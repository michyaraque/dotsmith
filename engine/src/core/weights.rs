//! Memory-mapped safetensors or GGUF: tensors are uploaded straight from the mapped file (no full read into RAM).
//! GGUF (ComfyUI-GGUF layout, e.g. Krea 2 Q3_K_M): plain F32/F16/BF16 tensors behave like safetensors ones; the
//! quantized ones come out of `qtensor` (still quantized, on the device) or `tensor` (dequantized to f32).
use anyhow::{Context, Result, bail};
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{DType, Device, Tensor};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static READ: AtomicU64 = AtomicU64::new(0);

/// Bytes handed out by every `Weights` so far (load progress: sample before and during a load).
pub fn bytes_read() -> u64 {
    READ.load(Ordering::Relaxed)
}

pub struct Entry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

pub struct Weights {
    map: Mmap,
    base: usize,
    pub entries: HashMap<String, Entry>,
    /// GGUF only: the ggml type of every tensor (quantized ones have Entry::dtype "GGUF").
    ggml: HashMap<String, GgmlDType>,
}

impl Weights {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut f = std::fs::File::open(path.as_ref()).with_context(|| format!("open {}", path.as_ref().display()))?;
        if path.as_ref().extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
            return Self::open_gguf(&mut f);
        }
        let map = unsafe { Mmap::map(&f)? };
        let n = u64::from_le_bytes(map[..8].try_into()?) as usize;
        let header: Value = serde_json::from_slice(&map[8..8 + n])?;
        let mut entries = HashMap::new();
        for (k, v) in header.as_object().context("bad safetensors header")? {
            if k == "__metadata__" {
                continue;
            }
            let off = v["data_offsets"].as_array().context("offsets")?;
            entries.insert(
                k.clone(),
                Entry {
                    dtype: v["dtype"].as_str().unwrap_or("").to_string(),
                    shape: v["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                    start: off[0].as_u64().unwrap() as usize,
                    end: off[1].as_u64().unwrap() as usize,
                },
            );
        }
        Ok(Self { map, base: 8 + n, entries, ggml: HashMap::new() })
    }

    fn open_gguf(f: &mut std::fs::File) -> Result<Self> {
        let content = gguf_file::Content::read(f)?;
        let map = unsafe { Mmap::map(&*f)? };
        let (mut entries, mut ggml) = (HashMap::new(), HashMap::new());
        for (name, info) in &content.tensor_infos {
            let mut shape = info.shape.dims().to_vec(); // candle already reversed ggml's order -> torch [out, in]
            // ComfyUI-GGUF stores some tensors flattened and keeps the real shape in the metadata
            if let Some(v) = content.metadata.get(&format!("comfy.gguf.orig_shape.{name}")) {
                shape = v
                    .to_vec()?
                    .iter()
                    .map(|x| x.to_u32().map(|d| d as usize).or_else(|_| x.to_i32().map(|d| d as usize)))
                    .collect::<candle_core::Result<_>>()?;
            }
            let n: usize = info.shape.elem_count();
            let size = n / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
            let start = (content.tensor_data_offset + info.offset) as usize;
            let dtype = match info.ggml_dtype {
                GgmlDType::F32 => "F32",
                GgmlDType::F16 => "F16",
                GgmlDType::BF16 => "BF16",
                _ => "GGUF",
            };
            entries.insert(name.clone(), Entry { dtype: dtype.into(), shape, start, end: start + size });
            ggml.insert(name.clone(), info.ggml_dtype);
        }
        Ok(Self { map, base: 0, entries, ggml })
    }

    /// True when `name` is a ggml-quantized tensor (GGUF Q*_K / Q*_0 ...), to be loaded with `qtensor`.
    pub fn is_quantized(&self, name: &str) -> bool {
        self.entries.get(name).is_some_and(|e| e.dtype == "GGUF")
    }

    /// A GGUF tensor kept in its ggml format on the device (shape [out, in] for a linear weight).
    pub fn qtensor(&self, name: &str, dev: &Device) -> Result<QTensor> {
        let (b, e) = self.bytes(name)?;
        let dt = *self.ggml.get(name).with_context(|| format!("{name}: not a GGUF tensor"))?;
        Ok(candle_core::quantized::ggml_file::qtensor_from_ggml(dt, b, e.shape.clone(), dev)?)
    }

    /// Total stored bytes of the tensors whose name passes `keep` (memory planning).
    pub fn bytes_where(&self, keep: impl Fn(&str) -> bool) -> u64 {
        self.entries.iter().filter(|(k, _)| keep(k)).map(|(_, e)| (e.end - e.start) as u64).sum()
    }

    /// Weight format of one layer's linears: "int8", "bf16", "gguf" or "other" (for reports).
    pub fn layer_format(&self, prefix: &str, linears: &[&str]) -> &'static str {
        match self.entries.get(&format!("{prefix}.{}.weight", linears.first().copied().unwrap_or(""))).map(|e| e.dtype.as_str()) {
            Some("I8") | Some("U8") => "int8",
            Some("BF16") => "bf16",
            Some("GGUF") => "gguf",
            _ => "other",
        }
    }

    /// Bytes of one layer's linears (`{prefix}.{linear}.weight`) when they can stream (core::stream): all int8
    /// (ComfyUI int8 / ConvRot) or all bf16 (plain checkpoints). None for anything else (GGUF, fp8, mixed).
    pub fn streamable_layer_bytes(&self, prefix: &str, linears: &[&str]) -> Option<u64> {
        let mut n = 0;
        let mut kind = None;
        for l in linears {
            let e = self.entries.get(&format!("{prefix}.{l}.weight"))?;
            let k = match e.dtype.as_str() {
                "I8" | "U8" => "int8",
                "BF16" => "bf16",
                _ => return None,
            };
            if kind.is_some_and(|x| x != k) {
                return None;
            }
            kind = Some(k);
            n += (e.end - e.start) as u64;
        }
        Some(n)
    }

    /// Absolute byte range of a tensor in the file (for readers that map the file themselves, e.g. core::stream).
    pub fn file_range(&self, name: &str) -> Result<(usize, usize)> {
        let e = self.entries.get(name).with_context(|| format!("missing tensor {name}"))?;
        Ok((self.base + e.start, self.base + e.end))
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// A tensor's bytes. Every weight read goes through here, so it is also where a load can be cancelled
    /// (core::ABORT).
    pub fn bytes(&self, name: &str) -> Result<(&[u8], &Entry)> {
        crate::core::check_abort()?;
        let e = self.entries.get(name).with_context(|| format!("missing tensor {name}"))?;
        READ.fetch_add((e.end - e.start) as u64, Ordering::Relaxed);
        Ok((&self.map[self.base + e.start..self.base + e.end], e))
    }

    /// A tensor in its stored dtype (bf16/f16/f32); I8 must go through `raw`.
    pub fn tensor(&self, name: &str, dev: &Device) -> Result<Tensor> {
        let (b, e) = self.bytes(name)?;
        let dt = match e.dtype.as_str() {
            "BF16" => DType::BF16,
            "F16" => DType::F16,
            "F32" => DType::F32,
            "U8" => DType::U8,
            "I64" => DType::I64,
            "GGUF" => return Ok(self.qtensor(name, dev)?.dequantize(dev)?),
            other => bail!("{name}: dtype {other} needs raw()"),
        };
        Ok(Tensor::from_raw_buffer(b, dt, &e.shape, dev)?)
    }

    /// A float tensor in the activation type (core::act_dtype): bf16 converted on the GPU, f32 widened on the CPU.
    pub fn tensor_act(&self, name: &str, dev: &Device) -> Result<Tensor> {
        match crate::core::act_dtype() {
            DType::F32 => self.tensor_f32(name, dev),
            dt => Ok(self.tensor(name, dev)?.to_dtype(dt)?),
        }
    }

    /// A float tensor as f32, widened on the CPU while it is read: cards before Ampere have no bf16 kernels to do it.
    pub fn tensor_f32(&self, name: &str, dev: &Device) -> Result<Tensor> {
        let (b, e) = self.bytes(name)?;
        let v: Vec<f32> = match e.dtype.as_str() {
            "F32" => return self.tensor(name, dev),
            "BF16" => b.as_chunks::<2>().0.iter().map(|c| half::bf16::from_le_bytes(*c).to_f32()).collect(),
            "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
            other => bail!("{name}: dtype {other} is not a float"),
        };
        Ok(Tensor::from_vec(v, e.shape.as_slice(), dev)?)
    }

    /// Bytes reinterpreted as `as_dtype` (int8 weights go in as u8 and the kernels read them signed).
    pub fn raw(&self, name: &str, as_dtype: DType, dev: &Device) -> Result<Tensor> {
        let (b, e) = self.bytes(name)?;
        Ok(Tensor::from_raw_buffer(b, as_dtype, &e.shape, dev)?)
    }

    /// A uint8 tensor holding JSON (ComfyUI `comfy_quant`, `comfy_attention.config`).
    pub fn json(&self, name: &str) -> Option<Value> {
        let (b, _) = self.bytes(name).ok()?;
        serde_json::from_slice(b).ok()
    }
}

//! The LoRAs in the user's library. Any `.safetensors` anywhere in it is listed, grouped by folder; its family is
//! read from its tensor names (the catalog's `lora_blocks`). Training output keeps its own grouping:
//!   <library>/<family id>/<version>/<run>/<run>[_<step>].safetensors
//! Nothing here loads a model; the UI lists what this returns and never hardcodes runs or bases.
//!
//! Optional metadata: `version.json` in a folder applies to every LoRA below it, `"loras": {"<file stem>": {...}}`
//! inside it to one file, and a sidecar `<file stem>.dotsmith.json` next to the .safetensors overrides both. Keys:
//! `base` (family id), `scale`, `background` ("#RRGGBB" or null), `trigger` (the LoRA's trigger word) and `walk`.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::model::Family;

/// How a version's LoRAs were trained (and so how their outputs must be generated and post-processed).
#[derive(Clone, Debug)]
pub struct VersionMeta {
    pub family: Family,
    /// Image pixels per sprite pixel (16 = x16, 8 = x8).
    pub scale: u32,
    /// Flat background colour the version was trained on (Krea: no alpha VAE -> magenta), None = real transparency.
    pub background: Option<[u8; 3]>,
    /// How this LoRA draws walk cycles, when it declares one (None = the built-in per-direction format).
    pub walk: Option<WalkFormat>,
    /// The word the LoRA was trained to respond to, put at the start of every prompt.
    pub trigger: Option<String>,
}

impl VersionMeta {
    /// The opening of every prompt: the trigger word, if any, then the medium.
    pub fn style(&self) -> String {
        match self.trigger.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
            Some(t) => format!("{t} pixel art"),
            None => "pixel art".into(),
        }
    }
}

/// A text that may come in several languages: "text" or {"en": "...", "es": "..."}.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Text {
    One(String),
    Many(BTreeMap<String, String>),
}

impl Text {
    /// The text in `lang` (two-letter code), else English, else any.
    pub fn get(&self, lang: &str) -> String {
        match self {
            Text::One(s) => s.clone(),
            Text::Many(m) => m.get(lang).or_else(|| m.get("en")).or_else(|| m.values().next()).cloned().unwrap_or_default(),
        }
    }
}

/// One choice for the template's `{kind}` (e.g. "female player character").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WalkKind {
    pub label: Text,
    pub text: String,
}

/// A walk-sheet LoRA's output format: one image holding a `cols` x `rows` grid of frames, one row per direction in
/// `row_dirs`; the missing directions are made from drawn ones (`derive`: "east": "transpose:south" | "mirror:west").
/// Cells are `width / scale / cols` x `height / scale / rows` sprite pixels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WalkFormat {
    /// image size the LoRA was trained at, px
    pub width: usize,
    pub height: usize,
    pub cols: usize,
    pub rows: usize,
    pub row_dirs: Vec<String>,
    pub derive: BTreeMap<String, String>,
    /// prompt with `{desc}` (the user's description) and `{kind}` (the chosen kind's text)
    pub prompt: String,
    pub kinds: Vec<WalkKind>,
    /// columns in playback order (e.g. [1, 0, 2, 0]: step, stand, step, stand); empty = 0..cols
    pub play: Vec<usize>,
    pub fps: u32,
}

impl Default for WalkFormat {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            cols: 1,
            rows: 1,
            row_dirs: vec![],
            derive: BTreeMap::new(),
            prompt: "{desc}".into(),
            kinds: vec![],
            play: vec![],
            fps: 8,
        }
    }
}

impl WalkFormat {
    pub fn cell(&self, scale: u32) -> (usize, usize) {
        let s = scale.max(1) as usize;
        (self.width / s / self.cols.max(1), self.height / s / self.rows.max(1))
    }

    pub fn play_order(&self) -> Vec<usize> {
        if self.play.is_empty() { (0..self.cols).collect() } else { self.play.iter().copied().filter(|&c| c < self.cols).collect() }
    }

    /// The prompt for `desc` with kind number `kind` (out of range = the first kind, none = empty).
    pub fn prompt_for(&self, desc: &str, kind: usize) -> String {
        let k = self.kinds.get(kind).or(self.kinds.first()).map(|k| k.text.as_str()).unwrap_or("");
        let p = self.prompt.replace("{kind}", k).replace("{desc}", desc);
        // a template without {kind} or an empty kind leaves ", ," behind
        p.replace(", ,", ",").replace(" ,", ",")
    }
}

pub struct Run {
    pub family: Family,
    pub version: PathBuf,
    pub dir: PathBuf,
    /// (label, path), newest step first: the step number, "final" for the un-numbered save, or the file name.
    pub ckpts: Vec<(String, PathBuf)>,
    pub meta: VersionMeta,
}

/// The folder whose metadata a LoRA follows: <family>/<version> in a training layout, else the file's own folder.
pub fn version_of(lora: &Path) -> PathBuf {
    let is_family = |p: &Path| p.file_name().and_then(|n| n.to_str()).and_then(Family::from_id).is_some();
    lora.ancestors().find(|p| p.parent().is_some_and(is_family)).map(Path::to_path_buf).unwrap_or_else(|| lora.parent().unwrap_or(lora).to_path_buf())
}

pub fn family_meta(family: Family) -> VersionMeta {
    let spec = family.spec();
    VersionMeta { family, scale: spec.scale, background: spec.background_rgb(), walk: None, trigger: None }
}

/// The tensor names of a safetensors file, from its header only.
fn tensor_names(path: &Path) -> Vec<String> {
    use std::io::Read;
    let read = || -> Option<Vec<String>> {
        let mut f = std::fs::File::open(path).ok()?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).ok()?;
        let n = u64::from_le_bytes(len);
        if n == 0 || n > 64 << 20 {
            return None;
        }
        let mut buf = vec![0u8; n as usize];
        f.read_exact(&mut buf).ok()?;
        let header: Value = serde_json::from_slice(&buf).ok()?;
        Some(header.as_object()?.keys().filter(|k| *k != "__metadata__").cloned().collect())
    };
    read().unwrap_or_default()
}

/// The family a LoRA file was trained for, from the module names it patches.
pub fn detect_family(lora: &Path) -> Option<Family> {
    let fams: Vec<(Family, Vec<String>)> = Family::ALL.into_iter().map(|f| (f, f.spec().lora_modules)).filter(|(_, m)| !m.is_empty()).collect();
    let names = tensor_names(lora);
    names.iter().find_map(|k| {
        let (key, kohya) = match k.strip_prefix("lora_unet_") {
            Some(r) => (r, true),
            None => (k.strip_prefix("diffusion_model.").or_else(|| k.strip_prefix("transformer.")).unwrap_or(k), false),
        };
        fams.iter()
            .find(|(_, mods)| {
                mods.iter().any(|m| {
                    let m = if kohya { m.replace('.', "_") } else { m.clone() };
                    module_matches(key, &m)
                })
            })
            .map(|(f, _)| *f)
    })
}

/// `key` starts with `pattern`, where `N` in the pattern stands for a block number.
fn module_matches(key: &str, pattern: &str) -> bool {
    let mut rest = key;
    let mut parts = pattern.split('N').peekable();
    while let Some(p) = parts.next() {
        let Some(r) = rest.strip_prefix(p) else { return false };
        rest = r;
        if parts.peek().is_some() {
            let digits = rest.chars().take_while(char::is_ascii_digit).count();
            if digits == 0 {
                return false;
            }
            rest = &rest[digits..];
        }
    }
    true
}

pub fn meta_for(lora: &Path, family: Family) -> VersionMeta {
    if lora.as_os_str().is_empty() { family_meta(family) } else { meta_of(lora) }
}

/// The version's metadata: version.json, else the family's defaults from the catalog.
pub fn meta_of_version(version: &Path) -> VersionMeta {
    version_meta(version, None)
}

/// `guess`: the family when neither version.json nor the folder names one.
fn version_meta(version: &Path, guess: Option<Family>) -> VersionMeta {
    let folder = version.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let v: Value = std::fs::read_to_string(version.join("version.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null);
    let family = v["base"].as_str().and_then(Family::from_id).or_else(|| Family::from_id(&folder)).or(guess).unwrap_or(Family::ALL[0]);
    let mut m = family_meta(family);
    apply(&mut m, &v);
    m
}

/// `scale`, `background` and `walk` of a metadata object onto `m` (missing keys keep what `m` has).
fn apply(m: &mut VersionMeta, v: &Value) {
    if let Some(s) = v["scale"].as_u64() {
        m.scale = s as u32;
    }
    match v.get("background") {
        Some(Value::String(s)) => m.background = crate::manifest::parse_hex(s),
        Some(Value::Null) => m.background = None,
        _ => {}
    }
    if let Some(w) = v.get("walk") {
        m.walk = serde_json::from_value(w.clone()).ok();
    }
    if let Some(t) = v["trigger"].as_str() {
        m.trigger = Some(t.to_string());
    }
}

/// A LoRA's metadata: its version's, then `version.json` "loras"."<stem>", then the `<stem>.dotsmith.json` sidecar.
pub fn meta_of(lora: &Path) -> VersionMeta {
    let version = version_of(lora);
    let mut m = version_meta(&version, detect_family(lora));
    let stem = lora.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let read = |p: PathBuf| -> Value { std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null) };
    let v = read(version.join("version.json"));
    if let Some(o) = v["loras"].get(&stem) {
        apply(&mut m, o);
    }
    apply(&mut m, &read(lora.with_file_name(format!("{stem}.dotsmith.json"))));
    m
}

/// Checkpoints of a run folder: `<run>.safetensors` ("final") and `<run>_<step>.safetensors`, newest step first; any
/// other .safetensors (a folder of downloaded LoRAs, e.g. Civitai files) is listed after them under its own file name.
pub fn checkpoints(run: &Path) -> Vec<(String, PathBuf)> {
    let name = run.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let mut out: Vec<(String, u64, PathBuf)> = std::fs::read_dir(run)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .filter_map(|p| {
            let stem = p.file_stem()?.to_string_lossy().to_string();
            if stem == name {
                Some(("final".to_string(), u64::MAX, p))
            } else {
                match stem.strip_prefix(&format!("{name}_")).and_then(|s| s.trim_start_matches('0').parse::<u64>().ok()) {
                    Some(step) => Some((step.to_string(), step, p)),
                    None if stem.starts_with("optimizer") => None,
                    None => Some((stem, 0, p)),
                }
            }
        })
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.into_iter().map(|(l, _, p)| (l, p)).collect()
}

/// Every LoRA in the library, grouped by folder (and family), newest first.
pub fn runs_in(library: &Path) -> Vec<Run> {
    let mut found = vec![];
    for family in Family::ALL {
        for v in std::fs::read_dir(library.join(family.id())).into_iter().flatten().flatten() {
            if !v.path().is_dir() || v.file_name() == "datasets" {
                continue;
            }
            for r in std::fs::read_dir(v.path()).into_iter().flatten().flatten() {
                let p = r.path();
                if !p.is_dir() {
                    continue;
                }
                let ck = checkpoints(&p);
                if !ck.is_empty() {
                    let meta = meta_of_version(&v.path());
                    found.push(Run { family: meta.family, version: v.path(), dir: p, ckpts: ck, meta });
                }
            }
        }
    }
    loose_runs(library, 0, &mut found);
    let newest = |r: &Run| r.ckpts.iter().filter_map(|(_, p)| p.metadata().ok()?.modified().ok()).max();
    found.sort_by_cached_key(|r| std::cmp::Reverse(newest(r)));
    found
}

/// LoRA files outside the training layout: every folder holding .safetensors is one run per family found in it.
fn loose_runs(dir: &Path, depth: usize, out: &mut Vec<Run>) {
    const SKIP: [&str; 3] = ["datasets", "samples", "optimizer"];
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut by_family: BTreeMap<&'static str, (Family, Vec<PathBuf>)> = BTreeMap::new();
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_lowercase();
        if p.is_dir() {
            let training_root = depth == 0 && Family::from_id(&name).is_some();
            if !training_root && depth < 4 && !name.starts_with('.') && !SKIP.contains(&name.as_str()) {
                loose_runs(&p, depth + 1, out);
            }
        } else if name.ends_with(".safetensors")
            && !name.starts_with("optimizer")
            && let Some(f) = detect_family(&p)
        {
            by_family.entry(f.id()).or_insert((f, vec![])).1.push(p);
        }
    }
    for (_, (family, files)) in by_family {
        let mut ckpts: Vec<(String, PathBuf)> =
            files.into_iter().map(|p| (p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(), p)).collect();
        ckpts.sort_by_key(|a| a.0.to_lowercase());
        let mut meta = version_meta(dir, Some(family));
        meta.family = family;
        out.push(Run { family, version: dir.to_path_buf(), dir: dir.to_path_buf(), ckpts, meta });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_found_by_its_family_folder_in_any_library() {
        let l = Path::new("E:/My LoRAs/krea2/v2/run_a/run_a_000001000.safetensors");
        assert_eq!(version_of(l), Path::new("E:/My LoRAs/krea2/v2"));
        let loose = Path::new("E:/Downloads/loras/pixel_icons.safetensors");
        assert_eq!(version_of(loose), Path::new("E:/Downloads/loras"));
    }

    fn fake_lora(path: &Path, keys: &[String]) {
        let header: serde_json::Map<String, Value> =
            keys.iter().map(|k| (k.clone(), serde_json::json!({"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}))).collect();
        let h = serde_json::to_vec(&header).unwrap();
        let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
        bytes.extend(h);
        bytes.extend([0u8, 0]);
        std::fs::write(path, bytes).unwrap();
    }

    /// DOTSMITH_LIBRARY=<folder>: list what a real library shows and how long the scan takes.
    #[test]
    fn real_library_if_given() {
        let Ok(lib) = std::env::var("DOTSMITH_LIBRARY") else { return };
        let t = std::time::Instant::now();
        let runs = runs_in(Path::new(&lib));
        println!("{} runs in {:.0} ms", runs.len(), t.elapsed().as_secs_f64() * 1e3);
        for r in &runs {
            println!("{:7} {:2} ckpts  detected {:?}  {}", r.family.id(), r.ckpts.len(), detect_family(&r.ckpts[0].1).map(|f| f.id()), r.dir.display());
        }
    }

    #[test]
    fn loose_files_are_listed_under_the_family_their_tensors_name() {
        let root = std::env::temp_dir().join(format!("dotsmith-loose-test-{}", std::process::id()));
        let dir = root.join("downloaded");
        std::fs::create_dir_all(&dir).unwrap();
        let fams: Vec<(Family, String)> = Family::ALL.into_iter().filter_map(|f| f.spec().lora_modules.first().map(|m| (f, m.replace('N', "7")))).collect();
        assert!(!fams.is_empty());
        for (f, m) in &fams {
            let kohya = dir.join("kohya.safetensors");
            fake_lora(&kohya, &[format!("lora_unet_{}.lora_down.weight", m.replace('.', "_"))]);
            assert_eq!(detect_family(&kohya), Some(*f));
            std::fs::remove_file(kohya).unwrap();
            fake_lora(&dir.join(format!("{}.safetensors", f.id())), &[format!("diffusion_model.{m}.lora_A.weight")]);
        }
        fake_lora(
            &dir.join("flux.safetensors"),
            &["transformer.transformer_blocks.0.attn.to_q.lora_A.weight".into(), "transformer.single_transformer_blocks.3.proj_out.lora_A.weight".into()],
        );
        assert_eq!(detect_family(&dir.join("flux.safetensors")), None);
        let runs = runs_in(&root);
        for (f, _) in &fams {
            let r = runs.iter().find(|r| r.family == *f).expect("a run per family");
            assert_eq!((r.dir.as_path(), r.ckpts.len()), (dir.as_path(), 1));
            assert_eq!(meta_of(&r.ckpts[0].1).family, *f);
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_sidecar_overrides_the_folder_for_one_lora() {
        let root = std::env::temp_dir().join(format!("dotsmith-cat-test-{}", std::process::id()));
        let run = root.join("krea2").join("external").join("pack");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(
            root.join("krea2/external/version.json"),
            r##"{"base": "krea2", "scale": 16, "background": "#FF00FF",
            "loras": {"b": {"scale": 6}}}"##,
        )
        .unwrap();
        std::fs::write(
            run.join("a.dotsmith.json"),
            r##"{"scale": 8, "background": "#808080", "walk": {"width": 1536, "height": 1024,
            "cols": 3, "rows": 2, "row_dirs": ["south", "north"], "derive": {"east": "transpose:south"},
            "prompt": "x, {kind}, {desc}, y", "kinds": [{"label": {"en": "Female", "es": "Mujer"}, "text": "female"}], "play": [1, 0, 2, 0]}}"##,
        )
        .unwrap();
        let a = meta_of(&run.join("a.safetensors"));
        assert_eq!((a.scale, a.background), (8, Some([128, 128, 128])));
        let w = a.walk.unwrap();
        assert_eq!(w.cell(a.scale), (64, 64));
        assert_eq!(w.prompt_for("archer", 0), "x, female, archer, y");
        assert_eq!(w.kinds[0].label.get("es"), "Mujer");
        let b = meta_of(&run.join("b.safetensors"));
        assert_eq!((b.scale, b.background, b.walk.is_none()), (6, Some([255, 0, 255]), true));
        let c = meta_of(&run.join("c.safetensors"));
        assert_eq!(c.scale, 16);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn no_lora_follows_the_family_defaults() {
        for f in Family::ALL {
            let m = meta_for(Path::new(""), f);
            assert_eq!((m.scale, m.background), (f.spec().scale, f.spec().background_rgb()));
        }
    }
}

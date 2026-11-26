//! The generator, in process: a worker thread owns the Rust engine (dotsmith_engine: every model family on candle + CUDA)
//! and runs the studio modes (jobs.rs). The UI sends commands and drains `poll()` on a timer; images are
//! decoded/scaled on the worker so the UI thread only uploads them.
use dotsmith_engine::catalog;
use dotsmith_engine::core::weights::bytes_read;
use dotsmith_engine::engine::Engine;
use dotsmith_engine::model::{Family, ModelKind, ModelPaths, Variant};
use image::RgbaImage;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::jobs;
use crate::pixels;
use crate::tr;

pub enum Msg {
    Event(Value),
    Preview(crate::aurora::Colors), // the live x0 estimate's colours (reduced here, off the UI thread)
    Result(pixels::Item),
    Prepared { source: PathBuf, sprite: RgbaImage },
    Exited(String),
}

/// What the worker is doing with models right now, read by the UI every frame (not through the event queue, so it is
/// current even while the worker is stuck inside a multi-second load).
#[derive(Clone, Default)]
pub struct Live {
    pub loading: Option<Loading>,
    /// A model choice sent while the worker was busy, not started yet: (sequence, label).
    pub queued: Option<(u64, String)>,
}

/// A model going onto the GPU: progress is the weight bytes read since `start` against `expect` (the bytes the last
/// load of the same kind read, else the size of its files).
#[derive(Clone)]
pub struct Loading {
    pub what: String,
    pub since: Instant,
    start: u64,
    expect: u64,
}

impl Loading {
    pub fn fraction(&self) -> Option<f32> {
        (self.expect > 0).then(|| (bytes_read().saturating_sub(self.start) as f64 / self.expect as f64).min(0.99) as f32)
    }
    /// Seconds left at the rate so far (None until enough has been read to tell).
    pub fn eta(&self) -> Option<f64> {
        let f = self.fraction()? as f64;
        let el = self.since.elapsed().as_secs_f64();
        (f > 0.05 && el > 0.5).then(|| el * (1. - f) / f)
    }
}

enum Cmd {
    State,
    /// `load`: also put it on the GPU when nothing is loaded yet (a loaded engine always switches).
    Select {
        base: String,
        lora: String,
        load: bool,
        seq: u64,
    },
    Load,
    Unload,
    Prepare(PathBuf),
    Generate(Value),
    Configure(Folders),
}

#[derive(Clone, Default)]
pub struct Folders {
    pub library: Option<PathBuf>,
    /// Where results go; `fallback` when unset.
    pub output: Option<PathBuf>,
    pub fallback: PathBuf,
}

pub struct Backend {
    tx: Mutex<Option<Sender<Cmd>>>,
    queue: Arc<Mutex<Vec<Msg>>>,
    live: Arc<Mutex<Live>>,
    seq: std::sync::atomic::AtomicU64,
    cancel: Arc<AtomicBool>,
    pub log: PathBuf,
}

struct Worker {
    folders: Folders,
    engine: Option<Engine>,
    base: String,
    lora: PathBuf,
    lora_scale: f64,
    phase: &'static str,
    queue: Arc<Mutex<Vec<Msg>>>,
    live: Arc<Mutex<Live>>,
    cancel: Arc<AtomicBool>,
}

impl Worker {
    fn push(&self, m: Msg) {
        self.queue.lock().unwrap().push(m);
    }
    fn event(&self, v: Value) {
        self.push(Msg::Event(v));
    }
    fn set_phase(&mut self, p: &'static str) {
        self.phase = p;
        self.event(json!({"ev": "phase", "phase": p}));
    }
    fn kind(&self) -> ModelKind {
        ModelKind::from_label(&self.base).unwrap_or(ModelKind::new(Family::ALL[0], Variant::Base))
    }
    fn lora_opt(&self) -> Option<String> {
        (!self.lora.as_os_str().is_empty()).then(|| self.lora.to_string_lossy().to_string())
    }
    fn runs(&self) -> Vec<catalog::Run> {
        self.folders.library.as_deref().map(catalog::runs_in).unwrap_or_default()
    }
    fn out_dir(&self) -> PathBuf {
        self.folders.output.clone().unwrap_or_else(|| self.folders.fallback.clone())
    }
    fn meta(&self) -> catalog::VersionMeta {
        catalog::meta_for(&self.lora, self.kind().family)
    }

    fn state(&self) {
        let version = jobs::version_of(&self.lora);
        let all = self.runs();
        // the catalog's own spelling of the current run / checkpoint (paths from last_model.json may differ in slashes)
        let same = |a: &Path, b: &Path| a.canonicalize().ok().zip(b.canonicalize().ok()).is_some_and(|(x, y)| x == y);
        let (cur_run, cur_lora) = all
            .iter()
            .find_map(|r| r.ckpts.iter().find(|c| same(&c.1, &self.lora)).map(|c| (r.dir.to_string_lossy().to_string(), c.1.to_string_lossy().to_string())))
            .unwrap_or_else(|| (self.lora.parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default(), self.lora.to_string_lossy().to_string()));
        let runs: Vec<Value> = all
            .iter()
            .map(|r| {
                let v = r.version.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                let n = r.dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                let short = if r.dir == r.version { String::new() } else { n };
                let newest = r.ckpts.iter().filter_map(|(_, p)| p.metadata().ok()?.modified().ok()).max();
                json!({"label": format!("{v} · {short}"), "run": r.dir.to_string_lossy(), "family": r.family.id(), "name": short,
                   "version": v, "scale": r.meta.scale, "age": newest.map(age).unwrap_or_default(), "magenta": r.meta.background.is_some(),
                   "ckpts": r.ckpts.iter().map(|(l, p)| json!({"label": l, "path": p.to_string_lossy()})).collect::<Vec<_>>()})
            })
            .collect();
        self.event(json!({
            "ev": "state", "bases": jobs::bases(),
            "families": Family::ALL.iter().map(|f| {
                let s = f.spec();
                let mut variants = vec![json!({"label": ModelKind::new(*f, Variant::Base).label(), "name": s.variants.base.name, "turbo": false, "steps": s.variants.base.steps})];
                if f.has_turbo() {
                    variants.push(json!({"label": ModelKind::new(*f, Variant::Turbo).label(), "name": s.variants.turbo.name, "turbo": true, "steps": s.variants.turbo.steps}));
                }
                json!({"id": f.id(), "label": s.label, "repo": s.repo, "scale": s.scale, "alpha": s.background.is_none(), "capabilities": s.capabilities,
                       "sizes": s.sizes, "modes": s.modes, "plain": s.prompt == "plain", "opaque": s.opaque, "variants": variants})
            }).collect::<Vec<_>>(),
            "catalog_error": dotsmith_engine::manifest::user_error(),
            "store": dotsmith_engine::manifest::store().into_iter().map(|s| json!({"repo": s.repo, "label": s.label})).collect::<Vec<_>>(),
            "loaded": self.engine.as_ref().map(|e| e.kind().label()), "base": self.base, "family": self.kind().family.id(), "family_label": self.kind().family.label(),
            "lora": cur_lora,
            "run": cur_run, "version": (!self.lora.as_os_str().is_empty()).then(|| version.file_name().map(|n| n.to_string_lossy().to_string())).flatten(),
            "scale": self.meta().scale, "walk": self.meta().walk, "out_dir": self.out_dir().to_string_lossy(), "phase": self.phase,
            "library": self.folders.library.as_ref().map(|l| l.to_string_lossy().to_string()),
            "gpu": match dotsmith_engine::gpu::probe() {
                Ok(g) => json!({"ok": true, "name": g.name, "vram_mb": g.total_mb, "compute": format!("{}.{}", g.compute.0, g.compute.1),
                                "low_vram": dotsmith_engine::core::low_vram(),
                                // where the loaded model's layer stacks went (empty when nothing is loaded)
                                "memory": if self.engine.is_some() { serde_json::to_value(dotsmith_engine::core::memory::report()).unwrap_or_default() } else { json!([]) }}),
                Err(e) => json!({"ok": false, "error": e.to_string()}),
            },
            "files": Family::ALL.iter().map(|f| (f.id().to_string(), Value::from(dotsmith_engine::locations::resolve(*f).into_iter().map(|c| json!({
                "role": c.role, "variant": c.variant, "precision": c.precision, "path": c.path.to_string_lossy(), "found": c.found,
                "dest": dotsmith_engine::locations::download_root().map(|r| r.join(&c.file).to_string_lossy().to_string()),
                "remote": c.source.map(|r| json!({"repo": r.repo, "path": r.path, "url": r.url()}))})).collect::<Vec<_>>()))).collect::<serde_json::Map<_, _>>(),
            "runs": runs, "cameras": jobs::cameras().iter().map(|c| c.0).collect::<Vec<_>>(), "turbo": self.kind().turbo(),
            "recent": load_recent().into_iter().map(|(b, l)| json!({"base": b, "lora": l.to_string_lossy(), "label": model_label(&b, &l),
                "family": ModelKind::from_label(&b).map(|k| k.family.id()).unwrap_or("")})).collect::<Vec<_>>(),
        }));
    }

    fn begin_load(&mut self, key: &str) {
        let kind = self.kind();
        let lora = self.lora.metadata().map(|m| m.len()).unwrap_or(0);
        let expect = load_bytes(key)
            .unwrap_or_else(|| ModelPaths::default_for(kind.family).map(|p| p.load_bytes(kind.variant, key.starts_with("dit:"))).unwrap_or(0) + lora);
        let what = model_label(&self.base, &self.lora);
        self.live.lock().unwrap().loading = Some(Loading { what, since: Instant::now(), start: bytes_read(), expect });
        self.set_phase("loading");
    }

    fn end_load(&mut self, key: &str, ok: bool) {
        let done = self.live.lock().unwrap().loading.take();
        if let (true, Some(l)) = (ok, done) {
            record_load(key, bytes_read().saturating_sub(l.start));
        }
    }

    /// Put the selected base + LoRA on the GPU: a fresh load, a DiT swap (base <-> turbo), a family reload, or only a
    /// LoRA swap. Returns whether weights were (re)loaded; the timing goes to the status line.
    fn ensure_engine(&mut self) -> anyhow::Result<bool> {
        let kind = self.kind();
        let lora = self.lora_opt();
        let label = model_label(&self.base, &self.lora);
        let reload = match self.engine.as_ref().map(|e| e.kind()) {
            None => Some(format!("full:{}", kind.label())),
            Some(c) if c == kind => None,
            Some(c) if c.family == kind.family => Some(format!("dit:{}", kind.label())),
            Some(_) => Some(format!("full:{}", kind.label())),
        };
        let scale = self.lora_scale;
        let Some(key) = reload else {
            self.engine.as_mut().unwrap().set_lora(lora.as_deref(), scale)?;
            return Ok(false);
        };
        self.begin_load(&key);
        let t = Instant::now();
        let r = match self.engine.as_mut() {
            None => Engine::load(kind).and_then(|mut e| e.set_lora(lora.as_deref(), scale).map(|_| e)).map(|e| self.engine = Some(e)),
            Some(e) => e.set_kind(kind).and_then(|_| e.set_lora(lora.as_deref(), scale).map(|_| ())),
        };
        self.end_load(&key, r.is_ok());
        if let Err(err) = r {
            self.engine = None; // whatever was on the GPU is gone or half replaced: start clean on the next load
            return Err(err);
        }
        self.event(json!({"ev": "toast", "level": "ok", "message": tr!(format!("{label} listo en GPU · {:.1} s", t.elapsed().as_secs_f64()), format!("{label} ready on the GPU · {:.1} s", t.elapsed().as_secs_f64()))}));
        self.state(); // the memory placement of the new load (GPU settings)
        Ok(true)
    }

    /// Report a model change and settle the phase from what is really on the GPU (an error leaves it "off", never a
    /// stale "ready").
    fn after_model_change(&mut self, r: anyhow::Result<bool>, what: &str) {
        match r {
            Ok(reloaded) => {
                push_recent(&self.base, &self.lora);
                if !reloaded {
                    self.event(json!({"ev": "toast", "level": "ok", "message": format!("{what} · {}", model_label(&self.base, &self.lora))}));
                }
            }
            Err(e) => self.event(json!({"ev": "toast", "level": "error",
                "message": tr!(format!("No se pudo cargar {}: {}", model_label(&self.base, &self.lora), explain(&e)), format!("Could not load {}: {}", model_label(&self.base, &self.lora), explain(&e)))})),
        }
        let p = if self.engine.is_some() { "ready" } else { "off" };
        self.set_phase(p);
    }

    fn run(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::State => self.state(),
            Cmd::Select { base, lora, load, seq } => {
                {
                    let mut l = self.live.lock().unwrap();
                    if l.queued.as_ref().is_some_and(|q| q.0 == seq) {
                        l.queued = None;
                    }
                }
                let Some(kind) = ModelKind::from_label(&base) else {
                    self.event(json!({"ev": "toast", "level": "error", "message": tr!(format!("Modelo base desconocido: {base}"), format!("Unknown base model: {base}"))}));
                    self.state();
                    return;
                };
                // a LoRA belongs to one family: keep it if it matches the base, else take that family's newest one
                let explicit_none = lora == "-";
                let lora = PathBuf::from(&lora);
                let lora = if explicit_none {
                    PathBuf::new()
                } else if lora.is_file() && catalog::meta_of(&lora).family == kind.family {
                    lora
                } else {
                    newest_lora(&self.runs(), kind.family).unwrap_or_default()
                };
                let changed_base = base != self.base;
                self.base = base;
                self.lora = lora;
                save_last(&self.base, &self.lora);
                if self.engine.is_some() || load {
                    let r = self.ensure_engine();
                    self.after_model_change(r, if changed_base { tr!("Modelo cambiado", "Model changed") } else { tr!("LoRA cambiado", "LoRA changed") });
                } else {
                    push_recent(&self.base, &self.lora);
                }
                self.state();
            }
            Cmd::Load => {
                let r = self.ensure_engine();
                self.after_model_change(r, tr!("Ya estaba en GPU", "Already on the GPU"));
                self.state();
            }
            Cmd::Unload => {
                self.engine = None;
                self.event(json!({"ev": "toast", "level": "info", "message": tr!("GPU liberada.", "GPU memory freed.")}));
                self.set_phase("off");
                self.state();
            }
            Cmd::Prepare(path) => {
                match jobs::prepared_sprite(&path, self.meta().scale) {
                    Ok(sprite) => self.push(Msg::Prepared { source: path, sprite }),
                    Err(e) => self.event(json!({"ev": "toast", "level": "error", "message": tr!(format!("No se pudo abrir la imagen: {e}"), format!("Could not open the image: {e}"))})),
                }
            }
            Cmd::Configure(f) => {
                self.folders = f;
                if !self.lora.as_os_str().is_empty() && !self.lora.is_file() {
                    self.lora = newest_lora(&self.runs(), self.kind().family).unwrap_or_default();
                    save_last(&self.base, &self.lora);
                    if let Some(e) = self.engine.as_mut() {
                        let l = (!self.lora.as_os_str().is_empty()).then(|| self.lora.to_string_lossy().to_string());
                        let _ = e.set_lora(l.as_deref(), self.lora_scale);
                    }
                }
                self.state();
            }
            Cmd::Generate(req) => {
                self.cancel.store(false, Ordering::SeqCst);
                self.lora_scale = req["lora_scale"].as_f64().unwrap_or(1.0).clamp(0., 2.);
                self.event(json!({"ev": "started", "id": req["id"], "mode": req["mode"]}));
                let res = self.ensure_engine().and_then(|_| {
                    self.set_phase("busy");
                    let queue = self.queue.clone();
                    let q2 = self.queue.clone();
                    let q3 = self.queue.clone();
                    let mut emit = move |v: Value| queue.lock().unwrap().push(Msg::Event(v));
                    let key = self.meta().background;
                    let mut preview = move |img: RgbaImage| {
                        q2.lock().unwrap().push(Msg::Preview(crate::aurora::colors(&img, key)));
                    };
                    let mut result = move |png: PathBuf, settings: Value| {
                        if let Some(item) = pixels::Item::load(png, settings) {
                            q3.lock().unwrap().push(Msg::Result(item));
                        }
                    };
                    let (out_dir, meta) = (self.out_dir(), self.meta());
                    let mut ctx = jobs::Ctx {
                        engine: self.engine.as_mut().unwrap(), lora: self.lora.clone(), lora_scale: self.lora_scale, base: self.base.clone(), out_dir, meta,
                        cancel: &self.cancel, emit: &mut emit, preview: &mut preview, result: &mut result,
                    };
                    jobs::generate(&mut ctx, &req)
                });
                match res {
                    Ok(msg) => {
                        let status = if self.cancel.load(Ordering::SeqCst) { "stopped" } else { "done" };
                        self.event(json!({"ev": "done", "id": req["id"], "status": status, "message": msg}));
                    }
                    Err(e) => {
                        crate::log::error(&format!("generate {}: {e:#}", req["mode"]));
                        self.event(json!({"ev": "done", "id": req["id"], "status": "error", "message": explain(&e)}))
                    }
                }
                let p = if self.engine.is_some() { "ready" } else { "off" };
                self.set_phase(p);
            }
        }
    }
}

/// The error as the user should read it: out of VRAM says what to change.
fn explain(e: &anyhow::Error) -> String {
    if e.chain().any(|c| c.is::<dotsmith_engine::engine::VramFull>()) {
        return tr!(
            "Generación detenida: la VRAM de la GPU se llenó y Windows empezó a usar la RAM (por eso se colgaba). Usa un tamaño o cuadrícula menor, o cierra otras aplicaciones que usen la GPU (navegador, juegos, Discord…).",
            "Generation stopped: the GPU's VRAM filled up and Windows started using system RAM (that is what froze it). Use a smaller size or grid, or close other apps using the GPU (browser, games, Discord…)."
        ).to_string();
    }
    let text = format!("{e:#}");
    if text.contains("OUT_OF_MEMORY") || text.to_lowercase().contains("out of memory") {
        return tr!(
            "La GPU se ha quedado sin memoria. Activa Ajustes > GPU y memoria > Modo de poca VRAM, usa un tamaño menor o cierra otras aplicaciones que usen la GPU.",
            "The GPU ran out of memory. Turn on Settings > GPU & memory > Low-VRAM mode, use a smaller size or close other apps using the GPU."
        )
        .to_string();
    }
    text
}

/// %APPDATA%/Dotsmith/last_model.json: the base + LoRA picked last, restored as the default on the next start.
fn last_path() -> Option<PathBuf> {
    crate::settings::config_dir().map(|d| d.join("last_model.json"))
}

fn save_last(base: &str, lora: &Path) {
    if let Some(p) = last_path() {
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        let _ = std::fs::write(p, json!({"base": base, "lora": lora.to_string_lossy()}).to_string());
    }
}

fn load_last() -> Option<(String, PathBuf)> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(last_path()?).ok()?).ok()?;
    let (base, lora) = (v["base"].as_str()?.to_string(), PathBuf::from(v["lora"].as_str()?));
    let kind = ModelKind::from_label(&base)?;
    (lora.as_os_str().is_empty() || (lora.is_file() && catalog::meta_of(&lora).family == kind.family)).then_some((base, lora))
}

fn app_file(name: &str) -> Option<PathBuf> {
    last_path().map(|p| p.with_file_name(name))
}

/// "Krea 2 Turbo · v1 paso 5500": a base + LoRA as people read it.
pub fn model_label(base: &str, lora: &Path) -> String {
    if lora.as_os_str().is_empty() {
        return format!("{base} · {}", tr!("sin LoRA", "no LoRA"));
    }
    let version = jobs::version_of(lora).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    format!("{base} · {version} {}", checkpoint_name(lora))
}

/// "step 5500", "final" (the run's un-numbered save) or the file name of a LoRA from elsewhere.
pub fn checkpoint_name(lora: &Path) -> String {
    let stem = lora.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let run = lora.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    match stem.strip_prefix(&format!("{run}_")).filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())) {
        Some(n) => format!("{} {}", tr!("paso", "step"), n.trim_start_matches('0')),
        None if stem == run => "final".into(),
        None => stem,
    }
}

/// %APPDATA%/Dotsmith/recent_models.json: the last base + LoRA pairs used, newest first (one-click switch in the picker).
fn load_recent() -> Vec<(String, PathBuf)> {
    let text = app_file("recent_models.json").and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
    let Ok(v) = serde_json::from_str::<Value>(&text) else { return vec![] };
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    let (b, l) = (x["base"].as_str()?.to_string(), PathBuf::from(x["lora"].as_str()?));
                    let kind = ModelKind::from_label(&b)?;
                    (l.as_os_str().is_empty() || (l.is_file() && catalog::meta_of(&l).family == kind.family)).then_some((b, l))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn push_recent(base: &str, lora: &Path) {
    let Some(p) = app_file("recent_models.json") else { return };
    let same = |l: &Path| l.canonicalize().ok().zip(lora.canonicalize().ok()).map_or(l == lora, |(a, b)| a == b);
    let mut all = load_recent();
    all.retain(|(b, l)| !(b == base && same(l)));
    all.insert(0, (base.to_string(), lora.to_path_buf()));
    all.truncate(6);
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    let v: Vec<Value> = all.iter().map(|(b, l)| json!({"base": b, "lora": l.to_string_lossy()})).collect();
    let _ = std::fs::write(p, Value::from(v).to_string());
}

/// %APPDATA%/Dotsmith/load_bytes.json: weight bytes the last load of each kind read ("full:<base>" / "dit:<base>").
fn load_times() -> serde_json::Map<String, Value> {
    app_file("load_bytes.json")
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

fn load_bytes(key: &str) -> Option<u64> {
    load_times().get(key).and_then(|v| v.as_u64()).filter(|b| *b > 0)
}

/// The weight bytes a load of `key` read (a LoRA, a streamed layer stack or another precision change what one reads).
fn record_load(key: &str, bytes: u64) {
    let Some(p) = app_file("load_bytes.json") else { return };
    let mut m = load_times();
    m.insert(key.to_string(), json!(bytes));
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    let _ = std::fs::write(p, Value::Object(m).to_string());
}

/// "hace 3 h" style age of a file time.
fn age(t: std::time::SystemTime) -> String {
    let s = t.elapsed().map(|d| d.as_secs()).unwrap_or(0);
    match s {
        0..=59 => tr!("ahora", "just now").into(),
        60..=3599 => tr!(format!("hace {} min", s / 60), format!("{} min ago", s / 60)),
        3600..=86_399 => tr!(format!("hace {} h", s / 3600), format!("{} h ago", s / 3600)),
        _ => tr!(format!("hace {} d", s / 86_400), format!("{} d ago", s / 86_400)),
    }
}

/// The newest checkpoint of a family's newest run (None: no LoRA trained for it yet -> the bare base model).
fn newest_lora(runs: &[catalog::Run], family: Family) -> Option<PathBuf> {
    runs.iter().find(|r| r.family == family).and_then(|r| r.ckpts.first().map(|c| c.1.clone()))
}

fn ready_family() -> Family {
    Family::ALL.into_iter().find(|f| dotsmith_engine::locations::resolve(*f).iter().all(|c| c.found || c.variant.is_some())).unwrap_or(Family::ALL[0])
}

impl Backend {
    pub fn start(folders: Folders) -> Result<Arc<Backend>, String> {
        let queue: Arc<Mutex<Vec<Msg>>> = Arc::default();
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel::<Cmd>();
        let runs = folders.library.as_deref().map(catalog::runs_in).unwrap_or_default();
        let (base, lora) = load_last().unwrap_or_else(|| {
            let f = ready_family();
            (ModelKind::new(f, Variant::Base).label(), newest_lora(&runs, f).unwrap_or_default())
        });
        let live: Arc<Mutex<Live>> = Arc::default();
        let mut w =
            Worker { folders, engine: None, base, lora, lora_scale: 1.0, phase: "off", queue: queue.clone(), live: live.clone(), cancel: cancel.clone() };
        let q = queue.clone();
        std::thread::Builder::new()
            .name("engine".into())
            .spawn(move || {
                w.state();
                // opt-in preload only (SPRITE_STUDIO_PRELOAD=1); by default the GPU stays free until the user loads or generates
                if std::env::var("SPRITE_STUDIO_PRELOAD").is_ok() {
                    let r = w.ensure_engine();
                    w.after_model_change(r, tr!("Ya estaba en GPU", "Already on the GPU"));
                    w.state();
                }
                let mut backlog: VecDeque<Cmd> = VecDeque::new();
                loop {
                    let cmd = match backlog.pop_front() {
                        Some(c) => c,
                        None => match rx.recv() {
                            Ok(c) => c,
                            Err(_) => break,
                        },
                    };
                    backlog.extend(rx.try_iter());
                    // clicking through several models while one loads only loads the last pick: a model choice (or Load)
                    // followed by another choice with no generation in between is skipped
                    let superseded = matches!(cmd, Cmd::Select { .. } | Cmd::Load)
                        && backlog.iter().take_while(|c| !matches!(c, Cmd::Generate(_))).any(|c| matches!(c, Cmd::Select { .. }));
                    if superseded {
                        continue;
                    }
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.run(cmd)));
                    if let Err(e) = r {
                        let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                        crate::log::error(&format!("engine panic: {msg}"));
                        w.event(json!({"ev": "done", "status": "error", "message": tr!(format!("Fallo interno: {msg}"), format!("Internal error: {msg}"))}));
                        w.set_phase(if w.engine.is_some() { "ready" } else { "off" });
                    }
                }
                q.lock().unwrap().push(Msg::Exited(String::new()));
            })
            .map_err(|e| e.to_string())?;
        let log = crate::log::path();
        Ok(Arc::new(Backend { tx: Mutex::new(Some(tx)), queue, live, seq: Default::default(), cancel, log }))
    }

    /// The same JSON commands the UI always sent; `stop` acts at once (atomic flag, checked every step).
    pub fn send(&self, v: Value) {
        let cmd = match v["cmd"].as_str() {
            Some("state") => Cmd::State,
            Some("select") => {
                let (base, lora) = (v["base"].as_str().unwrap_or("").to_string(), v["lora"].as_str().unwrap_or("").to_string());
                let seq = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
                let shown = if lora == "-" { PathBuf::new() } else { PathBuf::from(&lora) };
                self.live.lock().unwrap().queued = Some((seq, model_label(&base, &shown)));
                Cmd::Select { base, lora, load: v["load"].as_bool().unwrap_or(false), seq }
            }
            Some("load") => Cmd::Load,
            Some("unload") => Cmd::Unload,
            Some("prepare") => Cmd::Prepare(PathBuf::from(v["path"].as_str().unwrap_or(""))),
            Some("generate") => Cmd::Generate(v),
            Some("stop") => {
                self.cancel.store(true, Ordering::SeqCst);
                return;
            }
            _ => return,
        };
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            let _ = tx.send(cmd);
        }
    }

    pub fn configure(&self, f: Folders) {
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            let _ = tx.send(Cmd::Configure(f));
        }
    }

    /// The worker's model activity right now (a clone under a mutex the worker only holds for an instant).
    pub fn live(&self) -> Live {
        self.live.lock().unwrap().clone()
    }

    pub fn poll(&self) -> Vec<Msg> {
        std::mem::take(&mut *self.queue.lock().unwrap())
    }

    pub fn shutdown(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        self.tx.lock().unwrap().take(); // the worker ends after its current command; the GPU frees with the process
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End to end on this machine's GPU and settings: `cargo test --release generates_a_sprite -- --ignored`.
    #[test]
    #[ignore]
    fn generates_a_sprite() {
        let settings = crate::settings::Settings::load();
        settings.apply_to_engine();
        let out = std::env::temp_dir().join(format!("dotsmith-e2e-{}", std::process::id()));
        let b = Backend::start(Folders { library: settings.lora_library(), output: Some(out.clone()), fallback: out.clone() }).unwrap();
        b.send(json!({"cmd": "generate", "mode": "single", "prompt": "a red potion bottle", "seed": 7, "steps": 8,
                      "camera": "Normal", "width": 32, "height": 32, "opaque": false, "center": true, "mirror": false}));
        let started = Instant::now();
        let mut saved = None;
        while started.elapsed().as_secs() < 300 {
            for m in b.poll() {
                match m {
                    Msg::Result(item) => saved = Some(item.png),
                    Msg::Event(v) if v["ev"] == "done" => {
                        assert_eq!(v["status"], "done", "{}", v["message"]);
                        let png = saved.expect("a result before done");
                        assert!(png.starts_with(&out) && png.with_extension("json").is_file());
                        let _ = std::fs::remove_dir_all(&out);
                        return;
                    }
                    _ => {}
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        panic!("no result in 300 s");
    }

    /// "Imagen" on real pictures, kept for a look: IMG2SPRITE_CASES = a JSON list of {"source", "prompt", "out_size",
    /// "restyle", "strength"}, results in IMG2SPRITE_OUT. `cargo test --release image_to_sprite_cases -- --ignored`.
    #[test]
    #[ignore]
    fn image_to_sprite_cases() {
        let cases: Vec<Value> = serde_json::from_str(&std::env::var("IMG2SPRITE_CASES").unwrap()).unwrap();
        let out = PathBuf::from(std::env::var("IMG2SPRITE_OUT").unwrap());
        let settings = crate::settings::Settings::load();
        settings.apply_to_engine();
        let b = Backend::start(Folders { library: settings.lora_library(), output: Some(out.clone()), fallback: out.clone() }).unwrap();
        for case in cases {
            let mut req = json!({"cmd": "generate", "mode": "photo", "seed": 7, "steps": 8, "variants": 1, "guidance": 3.0, "matte": true, "camera": "Normal"});
            for (k, v) in case.as_object().unwrap() {
                req[k] = v.clone();
            }
            b.send(req);
            let started = Instant::now();
            'wait: loop {
                assert!(started.elapsed().as_secs() < 300, "no result in 300 s");
                for m in b.poll() {
                    if let Msg::Event(v) = m
                        && v["ev"] == "done"
                    {
                        println!("{} -> {} {}", case, v["status"], v["message"]);
                        break 'wait;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }
}

//! Model downloads from Hugging Face on their own thread: one file at a time into `<file>.part`, resumed with a Range
//! request after an interruption and renamed when complete.
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::tr;

#[derive(Clone)]
pub struct Item {
    pub name: String,
    pub url: String,
    pub dest: PathBuf,
    pub repo: String,
}

/// A file that finished in this session (the downloads panel lists them).
#[derive(Clone)]
pub struct Finished {
    pub name: String,
    pub bytes: u64,
    pub dest: PathBuf,
}

#[derive(Clone, Default)]
pub struct Progress {
    pub family: String,
    pub running: bool,
    pub file: String,
    pub index: usize,
    pub count: usize,
    pub done: u64,
    pub total: u64,
    /// The batch: (name, bytes) in download order.
    pub items: Vec<(String, u64)>,
    pub history: Vec<Finished>,
    /// Set once when the batch ends: Ok(files) or why it stopped. Taken by the UI.
    pub finished: Option<Result<usize, Failure>>,
}

/// Why a batch stopped.
#[derive(Clone, Debug)]
pub enum Failure {
    /// The repo is gated: its license has to be accepted on huggingface.co and a token added (the repo id).
    Access(String),
    Other(String),
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::Access(repo) => write!(
                f,
                "{}",
                tr!(
                    format!("Hugging Face pide acceso a {repo}: acepta su licencia en huggingface.co/{repo} y añade tu token en Ajustes."),
                    format!("Hugging Face needs access to {repo}: accept its license at huggingface.co/{repo} and add your token in Settings.")
                )
            ),
            Failure::Other(m) => write!(f, "{m}"),
        }
    }
}

impl From<String> for Failure {
    fn from(m: String) -> Self {
        Failure::Other(m)
    }
}

impl From<&str> for Failure {
    fn from(m: &str) -> Self {
        Failure::Other(m.to_string())
    }
}

#[derive(Default)]
pub struct Downloader {
    state: Arc<Mutex<Progress>>,
    cancel: Arc<AtomicBool>,
}

impl Downloader {
    pub fn progress(&self) -> Progress {
        self.state.lock().unwrap().clone()
    }

    pub fn take_finished(&self) -> Option<Result<usize, Failure>> {
        self.state.lock().unwrap().finished.take()
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    pub fn start(&self, family: String, items: Vec<Item>, token: Option<String>) {
        if self.state.lock().unwrap().running || items.is_empty() {
            return;
        }
        self.cancel.store(false, Ordering::SeqCst);
        {
            let mut s = self.state.lock().unwrap();
            let history = std::mem::take(&mut s.history);
            *s = Progress {
                family,
                running: true,
                count: items.len(),
                items: items.iter().map(|i| (i.name.clone(), 0)).collect(),
                history,
                ..Default::default()
            };
        }
        let (state, cancel) = (self.state.clone(), self.cancel.clone());
        std::thread::Builder::new()
            .name("download".into())
            .spawn(move || {
                let result = run(&items, token.as_deref(), &state, &cancel);
                if let Err(e) = &result {
                    crate::log::error(&format!("download: {e}"));
                }
                let mut s = state.lock().unwrap();
                s.running = false;
                s.finished = Some(result.map(|_| items.len()));
            })
            .ok();
    }
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().user_agent(concat!("Dotsmith/", env!("CARGO_PKG_VERSION"))).build()
}

fn request(agent: &ureq::Agent, method: &str, url: &str, token: Option<&str>) -> ureq::Request {
    let r = agent.request(method, url);
    match token {
        Some(t) => r.set("Authorization", &format!("Bearer {t}")),
        None => r,
    }
}

fn explain(e: ureq::Error, item: &Item) -> Failure {
    match e {
        ureq::Error::Status(401 | 403, _) => Failure::Access(item.repo.clone()),
        ureq::Error::Status(404, _) => tr!(format!("{} ya no está en Hugging Face.", item.name), format!("{} is no longer on Hugging Face.", item.name)).into(),
        ureq::Error::Status(code, _) => format!("{}: HTTP {code}", item.name).into(),
        ureq::Error::Transport(t) => tr!(format!("Sin conexión con Hugging Face: {t}"), format!("Could not reach Hugging Face: {t}")).into(),
    }
}

fn part_of(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

fn run(items: &[Item], token: Option<&str>, state: &Mutex<Progress>, cancel: &AtomicBool) -> Result<(), Failure> {
    let agent = agent();
    let mut sizes = vec![];
    for it in items {
        let len = request(&agent, "HEAD", &it.url, token)
            .call()
            .map_err(|e| explain(e, it))?
            .header("Content-Length")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        sizes.push(len);
    }
    state.lock().unwrap().items = items.iter().zip(&sizes).map(|(i, &n)| (i.name.clone(), n)).collect();
    let needed: u64 = items.iter().zip(&sizes).map(|(it, n)| n.saturating_sub(part_of(&it.dest).metadata().map(|m| m.len()).unwrap_or(0))).sum();
    if let Some(dir) = items.first().and_then(|it| it.dest.parent()) {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        if let Some(free) = free_space(dir).filter(|&free| free < needed + (512 << 20)) {
            return Err(tr!(
                format!("Falta espacio en disco: se necesitan {} y hay {} libres.", gb(needed), gb(free)),
                format!("Not enough disk space: {} needed, {} free.", gb(needed), gb(free))
            )
            .into());
        }
    }
    for (i, (it, &total)) in items.iter().zip(&sizes).enumerate() {
        {
            let mut s = state.lock().unwrap();
            s.file = it.name.clone();
            s.index = i;
            s.total = total;
            s.done = 0;
        }
        fetch(&agent, it, token, total, state, cancel)?;
        state.lock().unwrap().history.insert(0, Finished { name: it.name.clone(), bytes: total, dest: it.dest.clone() });
    }
    Ok(())
}

fn fetch(agent: &ureq::Agent, it: &Item, token: Option<&str>, total: u64, state: &Mutex<Progress>, cancel: &AtomicBool) -> Result<(), Failure> {
    std::fs::create_dir_all(it.dest.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?;
    let part = part_of(&it.dest);
    let mut have = part.metadata().map(|m| m.len()).unwrap_or(0);
    if total > 0 && have > total {
        have = 0;
    }
    let mut req = request(agent, "GET", &it.url, token);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = req.call().map_err(|e| explain(e, it))?;
    let resumed = have > 0 && resp.status() == 206;
    let mut file = std::fs::OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part).map_err(|e| e.to_string())?;
    let mut done = if resumed { have } else { 0 };
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err(tr!("Descarga cancelada.", "Download cancelled.").into());
        }
        let n = reader.read(&mut buf).map_err(|e| tr!(format!("La descarga se cortó: {e}"), format!("The download was interrupted: {e}")))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        done += n as u64;
        state.lock().unwrap().done = done;
    }
    file.flush().map_err(|e| e.to_string())?;
    drop(file);
    if total > 0 && done != total {
        return Err(tr!(
            format!("{} quedó incompleto; vuelve a descargar para reanudar.", it.name),
            format!("{} is incomplete; download again to resume.", it.name)
        )
        .into());
    }
    Ok(std::fs::rename(&part, &it.dest).map_err(|e| e.to_string())?)
}

pub fn gb(bytes: u64) -> String {
    if bytes >= 1 << 30 { format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64) } else { format!("{:.0} MB", bytes as f64 / (1u64 << 20) as f64) }
}

fn free_space(dir: &Path) -> Option<u64> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        unsafe extern "system" {
            fn GetDiskFreeSpaceExW(dir: *const u16, free: *mut u64, total: *mut u64, total_free: *mut u64) -> i32;
        }
        let w: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let mut free = 0u64;
        (unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) } != 0).then_some(free)
    }
    #[cfg(not(windows))]
    {
        let _ = dir;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Downloads a small public file end to end: `cargo test --release downloads_a_file -- --ignored`.
    #[test]
    #[ignore]
    fn downloads_a_file() {
        let dir = std::env::temp_dir().join(format!("dotsmith-dl-{}", std::process::id()));
        let dest = dir.join("tokenizer.json");
        let item = Item {
            name: "tokenizer.json".into(),
            url: "https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/main/processor/tokenizer.json".into(),
            dest: dest.clone(),
            repo: "Qwen/Qwen-Image-2.1".into(),
        };
        let state = Mutex::new(Progress::default());
        run(&[item], None, &state, &AtomicBool::new(false)).unwrap();
        assert!(dest.metadata().unwrap().len() > 1_000_000);
        assert!(!part_of(&dest).exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}

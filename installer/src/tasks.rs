//! Install and uninstall, run on a worker thread; the window reads `Progress` to draw them.
use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::payload::Payload;
use crate::system;

/// The files an install put in its folder, one relative path per line (what the uninstall removes).
pub const LIST: &str = "uninstall.lst";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    Previous,
    Files,
    Shortcuts,
    Register,
}

#[derive(Default)]
pub struct Progress {
    pub step: Option<Step>,
    pub done: u64,
    pub total: u64,
    pub file: String,
    /// set once the task ends
    pub result: Option<Result<(), String>>,
}

pub type Shared = Arc<Mutex<Progress>>;

fn set(p: &Shared, step: Step) {
    p.lock().unwrap().step = Some(step);
}

pub struct Install {
    pub dir: PathBuf,
    pub desktop: bool,
    pub version: String,
}

pub fn spawn(p: Shared, job: impl FnOnce(&Shared) -> Result<()> + Send + 'static) {
    std::thread::spawn(move || {
        let r = job(&p).map_err(|e| format!("{e:#}"));
        p.lock().unwrap().result = Some(r);
    });
}

pub fn install(job: &Install, payload: &Payload, p: &Shared) -> Result<()> {
    if let Some(old) = system::legacy() {
        set(p, Step::Previous);
        old.remove()?;
    }
    set(p, Step::Files);
    let exe = job.dir.join(system::EXE);
    if system::in_use(&exe) {
        bail!(in_use_message());
    }
    std::fs::create_dir_all(&job.dir).with_context(|| format!("create {}", job.dir.display()))?;
    let before = read_list(&job.dir);
    p.lock().unwrap().total = payload.raw;
    let mut files = payload.extract(&job.dir, |done, file| {
        let mut g = p.lock().unwrap();
        g.done = done;
        if g.file != file {
            g.file = file.to_string();
        }
    })?;
    payload.write_stub(&job.dir.join(system::UNINSTALLER))?;
    files.extend([system::UNINSTALLER.to_string(), LIST.to_string()]);
    // an update: what the old version had and this one does not goes
    let keep: HashSet<&str> = files.iter().map(String::as_str).collect();
    remove_files(&job.dir, before.iter().filter(|f| !keep.contains(f.as_str())));
    std::fs::write(job.dir.join(LIST), files.join("\n"))?;

    set(p, Step::Shortcuts);
    if let Some(l) = system::start_menu_link() {
        system::create_link(&l, &exe)?;
    }
    if let Some(l) = system::desktop_link() {
        if job.desktop {
            system::create_link(&l, &exe)?;
        } else {
            let _ = std::fs::remove_file(l);
        }
    }

    set(p, Step::Register);
    system::register(&job.dir, &job.version, payload.raw + payload.stub)?;
    Ok(())
}

pub struct Uninstall {
    pub dir: PathBuf,
    pub models: bool,
    pub settings: bool,
}

pub fn uninstall(job: &Uninstall, p: &Shared) -> Result<()> {
    set(p, Step::Files);
    if system::in_use(&job.dir.join(system::EXE)) {
        bail!(in_use_message());
    }
    remove_files(&job.dir, read_list(&job.dir).iter());
    let _ = std::fs::remove_dir(&job.dir); // only when empty: files the user added there stay

    set(p, Step::Shortcuts);
    for l in [system::start_menu_link(), system::desktop_link()].into_iter().flatten() {
        let _ = std::fs::remove_file(l);
    }
    if job.models
        && let Some(m) = system::models_dir()
    {
        let _ = std::fs::remove_dir_all(m);
    }
    if job.settings {
        for d in system::settings_dirs() {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    set(p, Step::Register);
    system::unregister();
    Ok(())
}

fn in_use_message() -> String {
    crate::tr!("Dotsmith está abierto. Ciérralo y vuelve a intentarlo.", "Dotsmith is open. Close it and try again.").into()
}

fn read_list(dir: &Path) -> Vec<String> {
    std::fs::read_to_string(dir.join(LIST)).map(|s| s.lines().filter(|l| !l.is_empty()).map(String::from).collect()).unwrap_or_default()
}

/// Remove the listed files and the folders they leave empty (never anything outside `dir`).
fn remove_files<'a>(dir: &Path, files: impl Iterator<Item = &'a String>) {
    let mut parents = HashSet::new();
    for f in files {
        if f.split(['/', '\\']).any(|c| c == ".." || c.is_empty()) {
            continue;
        }
        let p = dir.join(f);
        let _ = std::fs::remove_file(&p);
        let mut d = p.parent();
        while let Some(x) = d.filter(|x| *x != dir) {
            parents.insert(x.to_path_buf());
            d = x.parent();
        }
    }
    let mut parents: Vec<PathBuf> = parents.into_iter().collect();
    parents.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    for d in parents {
        let _ = std::fs::remove_dir(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_update_removes_only_the_files_it_listed() {
        let dir = std::env::temp_dir().join(format!("dotsmith-setup-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("licenses")).unwrap();
        for f in ["old.dll", "licenses/old.txt", "mine.txt"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let listed = ["old.dll".to_string(), "licenses/old.txt".to_string(), "../outside.txt".to_string()];
        remove_files(&dir, listed.iter());
        assert!(!dir.join("old.dll").exists());
        assert!(!dir.join("licenses").exists(), "an emptied folder goes");
        assert!(dir.join("mine.txt").exists(), "a file the installer did not put there stays");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

//! Notices LoRAs and model files added, removed or replaced on disk while the app runs, so the lists update without
//! a restart or a manual refresh.
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use dotsmith_engine::model::Family;

const EVERY: Duration = Duration::from_secs(2);

/// Look at the LoRA library (the folder in `library`, which settings may change) and the model files every 2 s and
/// call `changed` when they differ from the last look and have stopped changing: a copy in progress fires once, when it
/// ends. Stops when `changed` returns false (the app is gone).
pub fn spawn(library: Arc<Mutex<Option<PathBuf>>>, changed: impl Fn() -> bool + Send + 'static) {
    let _ = std::thread::Builder::new().name("watch".into()).spawn(move || {
        let look = || snapshot(library.lock().unwrap().clone().as_deref());
        let (mut seen, mut pending) = (look(), None);
        loop {
            std::thread::sleep(EVERY);
            let now = look();
            if now == seen {
                pending = None;
            } else if pending == Some(now) {
                seen = now;
                pending = None;
                if !changed() {
                    return;
                }
            } else {
                pending = Some(now);
            }
        }
    });
}

fn snapshot(library: Option<&Path>) -> u64 {
    let mut files = vec![];
    if let Some(l) = library {
        walk(l, 0, &mut files);
    }
    let mut h = DefaultHasher::new();
    files.sort();
    files.hash(&mut h);
    // the model files of every family where the catalog looks for them: found or not, and their size
    for f in Family::ALL {
        for c in dotsmith_engine::locations::resolve(f) {
            (c.path.metadata().map(|m| m.len()).ok(), c.found).hash(&mut h);
        }
    }
    h.finish()
}

/// Every .safetensors under `dir` the LoRA catalog would list (same depth and skipped folders), with size and date.
fn walk(dir: &Path, depth: usize, out: &mut Vec<(PathBuf, u64, Option<SystemTime>)>) {
    const SKIP: [&str; 3] = ["datasets", "samples", "optimizer"];
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().to_lowercase();
        let Ok(meta) = e.metadata() else { continue };
        if meta.is_dir() {
            if depth < 5 && !name.starts_with('.') && !SKIP.contains(&name.as_str()) {
                walk(&e.path(), depth + 1, out);
            }
        } else if name.ends_with(".safetensors") {
            out.push((e.path(), meta.len(), meta.modified().ok()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_or_grown_lora_changes_the_snapshot() {
        let dir = std::env::temp_dir().join(format!("dotsmith-watch-{}", std::process::id()));
        let run = dir.join("krea2").join("v1").join("run");
        std::fs::create_dir_all(dir.join("samples")).unwrap();
        std::fs::create_dir_all(&run).unwrap();
        let empty = snapshot(Some(&dir));
        std::fs::write(dir.join("samples").join("ignored.safetensors"), b"x").unwrap();
        assert_eq!(snapshot(Some(&dir)), empty, "samples/ is not part of the library");
        std::fs::write(run.join("a.safetensors"), b"x").unwrap();
        let one = snapshot(Some(&dir));
        assert_ne!(one, empty);
        std::fs::write(run.join("a.safetensors"), b"xx").unwrap();
        assert_ne!(snapshot(Some(&dir)), one, "a file still being copied keeps changing");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

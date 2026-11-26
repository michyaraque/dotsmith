//! Where each family's base-model files are on this machine: configured folders, the environment, the default folders,
//! then the extra places the catalog (manifest.rs) lists.
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::manifest::{self, Source};
use crate::model::Family;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Locations {
    pub models_dir: Option<PathBuf>,
    pub hf_dirs: Vec<PathBuf>,
}

static CURRENT: RwLock<Option<Locations>> = RwLock::new(None);

pub fn set(l: Locations) {
    *CURRENT.write().unwrap() = Some(l);
}

pub fn get() -> Locations {
    CURRENT.read().unwrap().clone().unwrap_or_default()
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf))
}

pub fn default_models_dir() -> Option<PathBuf> {
    if let Some(l) = env_path("LOCALAPPDATA") {
        return Some(l.join("Dotsmith").join("models"));
    }
    env_path("HOME").map(|h| h.join(".local/share/dotsmith/models"))
}

pub fn model_roots() -> Vec<PathBuf> {
    let candidates = [
        get().models_dir,
        env_path("DOTSMITH_MODELS"),
        env_path("AI_TOOLKIT").map(|t| t.join("models")),
        default_models_dir(),
        exe_dir().map(|d| d.join("models")),
    ];
    let mut out: Vec<PathBuf> = vec![];
    for p in candidates.into_iter().flatten() {
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

fn hf_dirs(name: &str) -> Vec<PathBuf> {
    let mut out = get().hf_dirs;
    for r in model_roots() {
        out.push(r.join(name));
        out.push(r.join("hf").join(name));
    }
    out
}

fn hub_snapshots(repo: &str) -> Vec<PathBuf> {
    let hub = env_path("HF_HUB_CACHE")
        .or_else(|| env_path("HF_HOME").map(|h| h.join("hub")))
        .or_else(|| env_path("USERPROFILE").or_else(|| env_path("HOME")).map(|h| h.join(".cache/huggingface/hub")));
    let Some(hub) = hub else { return vec![] };
    let dir = hub.join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
    std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).collect()
}

/// `{hf:Name}/rest`, `{hub:owner/repo}/rest` or `{env:VAR}/rest` as the concrete paths it stands for.
fn expand(pattern: &str) -> Vec<PathBuf> {
    let Some((head, rest)) = pattern.strip_prefix('{').and_then(|p| p.split_once('}')) else {
        return vec![PathBuf::from(pattern)];
    };
    let rest = rest.trim_start_matches(['/', '\\']);
    let bases = match head.split_once(':') {
        Some(("hf", name)) => hf_dirs(name),
        Some(("hub", repo)) => hub_snapshots(repo),
        Some(("env", var)) => env_path(var).into_iter().collect(),
        _ => vec![],
    };
    bases.into_iter().map(|b| b.join(rest)).collect()
}

#[derive(Clone, Debug)]
pub struct FileCheck {
    pub role: String,
    /// "turbo" when only that variant needs the file.
    pub variant: Option<String>,
    /// Set for an optional alternative in another precision (manifest::FileSpec::precision).
    pub precision: Option<String>,
    pub path: PathBuf,
    pub found: bool,
    /// Path inside the models folder (where a download goes).
    pub file: String,
    pub source: Option<Source>,
}

/// Where downloads go: the configured models folder, else `%LOCALAPPDATA%\Dotsmith\models`.
pub fn download_root() -> Option<PathBuf> {
    get().models_dir.or_else(default_models_dir)
}

/// Every file a family needs, from the catalog (manifest.rs): the first candidate on disk, or where it is expected.
pub fn resolve(family: Family) -> Vec<FileCheck> {
    manifest::family(family)
        .files
        .into_iter()
        .map(|f| {
            let direct = f.env.as_deref().and_then(env_path);
            let mut cands: Vec<PathBuf> = direct.clone().into_iter().collect();
            cands.extend(model_roots().into_iter().map(|r| r.join(&f.file)));
            cands.extend(f.also.iter().flat_map(|a| expand(a)));
            let (path, found) = match direct {
                Some(d) => (d.clone(), d.is_file()),
                None => match cands.iter().find(|p| p.is_file()) {
                    Some(p) => (p.clone(), true),
                    None => (cands.into_iter().next().unwrap_or_default(), false),
                },
            };
            FileCheck { role: f.role, variant: f.variant, precision: f.precision, path, found, file: f.file, source: f.source }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_folder_comes_first_and_missing_files_point_there() {
        let family = Family::ALL[0];
        let spec = family.spec();
        let present = &spec.files[0].file;
        let dir = std::env::temp_dir().join(format!("dotsmith-loc-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(present).parent().unwrap()).unwrap();
        std::fs::write(dir.join(present), b"x").unwrap();
        set(Locations { models_dir: Some(dir.clone()), hf_dirs: vec![] });
        assert_eq!(model_roots()[0], dir);
        let files = resolve(family);
        assert_eq!(files.len(), spec.files.len());
        assert!(files[0].found && files[0].path == dir.join(present));
        for f in files.iter().filter(|f| !f.found) {
            assert!(f.path.starts_with(&dir), "{} expected under the configured folder", f.file);
        }
        set(Locations::default());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn catalog_patterns_expand() {
        assert_eq!(expand("plain/path.json"), vec![PathBuf::from("plain/path.json")]);
        assert!(expand("{env:DOTSMITH_SURELY_UNSET_VAR}/x").is_empty());
    }
}

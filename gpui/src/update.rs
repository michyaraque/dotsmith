//! New versions from the GitHub releases: a check at start and every 12 hours, the installer for this build (standard
//! or legacy) downloaded and checked against the release's SHA256SUMS.txt, then handed to Dotsmith Setup
//! (`--update --wait <pid>`), which waits for the app to close, installs over it and opens it again.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const REPO: &str = "michyaraque/dotsmith";
const EVERY: Duration = Duration::from_secs(12 * 3600);

#[derive(Clone, Debug, PartialEq)]
pub struct Release {
    pub version: String,
    /// the release page (notes)
    pub page: String,
    setup: Option<Asset>,
    sums: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct Asset {
    name: String,
    url: String,
    size: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    Idle,
    Checking,
    UpToDate,
    Available(Release),
    Downloading { release: Release, done: u64, total: u64 },
    Ready { release: Release, setup: PathBuf },
    Failed(String),
}

impl State {
    /// The newer release this state is about, if any.
    pub fn release(&self) -> Option<&Release> {
        match self {
            State::Available(r) | State::Downloading { release: r, .. } | State::Ready { release: r, .. } => Some(r),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct Updater {
    state: Arc<Mutex<State>>,
    /// bumped on every state change, so the window redraws only then
    pub seq: Arc<AtomicU64>,
    /// look on its own at start and every 12 hours (Settings > About)
    auto: Arc<AtomicBool>,
}

impl Updater {
    /// `auto`: check shortly after start (off the start-up path) and every 12 hours.
    pub fn start(auto: bool) -> Self {
        let u = Updater { state: Arc::new(Mutex::new(State::Idle)), seq: Arc::default(), auto: Arc::new(AtomicBool::new(auto)) };
        let me = u.clone();
        std::thread::spawn(move || {
            // the installer of a past update (the one running right now, if any, stays until next time)
            let _ = std::fs::remove_dir_all(download_dir());
            std::thread::sleep(Duration::from_secs(5));
            loop {
                if me.auto.load(Ordering::Relaxed) {
                    me.check_now();
                }
                std::thread::sleep(EVERY);
            }
        });
        u
    }

    pub fn set_auto(&self, on: bool) {
        self.auto.store(on, Ordering::Relaxed);
    }

    pub fn state(&self) -> State {
        self.state.lock().unwrap().clone()
    }

    fn set(&self, s: State) {
        *self.state.lock().unwrap() = s;
        self.seq.fetch_add(1, Ordering::Relaxed);
    }

    /// Look for a newer release now, on a worker thread.
    pub fn check(&self) {
        let me = self.clone();
        std::thread::spawn(move || me.check_now());
    }

    fn check_now(&self) {
        // a download in progress or finished stays as it is
        if matches!(self.state(), State::Downloading { .. } | State::Ready { .. }) {
            return;
        }
        self.set(State::Checking);
        self.set(match latest() {
            Ok(Some(r)) => State::Available(r),
            Ok(None) => State::UpToDate,
            Err(e) => {
                crate::log::error(&format!("update check: {e}"));
                State::Failed(crate::tr!("No se pudo comprobar si hay actualizaciones.", "Could not check for updates.").into())
            }
        });
    }

    /// Download the new installer and check it; afterwards the state is Ready (or Failed).
    pub fn download(&self) {
        let Some(release) = self.state().release().cloned() else { return };
        if matches!(self.state(), State::Downloading { .. } | State::Ready { .. }) {
            return;
        }
        let me = self.clone();
        std::thread::spawn(move || {
            let r = me.fetch(&release);
            me.set(match r {
                Ok(setup) => State::Ready { release, setup },
                Err(e) => {
                    crate::log::error(&format!("update download: {e}"));
                    State::Failed(crate::tr!(format!("No se pudo descargar la actualización: {e}"), format!("The update could not be downloaded: {e}")))
                }
            });
        });
    }

    fn fetch(&self, release: &Release) -> Result<PathBuf, String> {
        let asset = release.setup.clone().ok_or_else(|| crate::tr!("la versión no trae instalador", "the release has no installer").to_string())?;
        let expected = match &release.sums {
            Some(url) => sum_for(&get(url)?.into_string().map_err(|e| e.to_string())?, &asset.name),
            None => None,
        }
        .ok_or_else(|| crate::tr!("la versión no trae sumas de comprobación", "the release has no checksums").to_string())?;
        let dir = download_dir();
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(&asset.name);
        let part = path.with_extension("part");
        let mut body = get(&asset.url)?.into_reader();
        let mut out = std::fs::File::create(&part).map_err(|e| e.to_string())?;
        let (mut hash, mut done, mut buf) = (Sha256::new(), 0u64, vec![0u8; 1 << 20]);
        self.set(State::Downloading { release: release.clone(), done: 0, total: asset.size });
        loop {
            let n = body.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            hash.update(&buf[..n]);
            done += n as u64;
            if let State::Downloading { done: d, .. } = &mut *self.state.lock().unwrap() {
                *d = done;
            }
            self.seq.fetch_add(1, Ordering::Relaxed);
        }
        drop(out);
        let got: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if got != expected {
            let _ = std::fs::remove_file(&part);
            return Err(crate::tr!("el archivo descargado no coincide con su suma", "the downloaded file does not match its checksum").into());
        }
        std::fs::rename(&part, &path).map_err(|e| e.to_string())?;
        Ok(path)
    }

    /// Start the downloaded installer in update mode; the caller closes the app right after, and the installer opens
    /// it again when it is done.
    pub fn install(&self) -> Result<(), String> {
        let State::Ready { setup, .. } = self.state() else { return Err("no update downloaded".into()) };
        std::process::Command::new(&setup).args(["--update", "--wait", &std::process::id().to_string()]).spawn().map(|_| ()).map_err(|e| e.to_string())
    }
}

fn download_dir() -> PathBuf {
    std::env::temp_dir().join("dotsmith-update")
}

/// Percent of a download, 0 while its size is unknown.
pub fn percent(done: u64, total: u64) -> u64 {
    (done * 100).checked_div(total).unwrap_or(0)
}

/// Installed by Dotsmith Setup (it can update itself); a portable copy (the zip) gets the release page instead.
pub fn installed() -> bool {
    std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("uninstall.lst").is_file())).unwrap_or(false)
}

/// This build's version; DOTSMITH_UPDATE_FROM pretends another one (to try an update from a development build).
pub fn current() -> String {
    std::env::var("DOTSMITH_UPDATE_FROM").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

/// The installer this build updates with: the legacy package for cards before RTX 30, the standard one otherwise.
fn setup_name(version: &str) -> String {
    let flavour = if dotsmith_engine::gpu::MIN_COMPUTE < (8, 0) { "-legacy" } else { "" };
    format!("Dotsmith-{version}{flavour}-setup.exe")
}

fn get(url: &str) -> Result<ureq::Response, String> {
    ureq::get(url)
        .set("User-Agent", concat!("Dotsmith/", env!("CARGO_PKG_VERSION")))
        .set("Accept", "application/vnd.github+json")
        .timeout(Duration::from_secs(30))
        .call()
        .map_err(|e| e.to_string())
}

/// The latest published release, when it is newer than this build.
fn latest() -> Result<Option<Release>, String> {
    let v: Value = get(&format!("https://api.github.com/repos/{REPO}/releases/latest"))?.into_json().map_err(|e| e.to_string())?;
    Ok(parse_release(&v).filter(|r| newer(&r.version, &current())))
}

fn parse_release(v: &Value) -> Option<Release> {
    let version = v["tag_name"].as_str()?.trim_start_matches('v').to_string();
    let assets = v["assets"].as_array().cloned().unwrap_or_default();
    let find = |name: &str| assets.iter().find(|a| a["name"].as_str() == Some(name));
    let setup = find(&setup_name(&version)).map(|a| Asset {
        name: a["name"].as_str().unwrap_or_default().to_string(),
        url: a["browser_download_url"].as_str().unwrap_or_default().to_string(),
        size: a["size"].as_u64().unwrap_or(0),
    });
    let sums = find("SHA256SUMS.txt").and_then(|a| a["browser_download_url"].as_str()).map(String::from);
    Some(Release { version, page: v["html_url"].as_str().unwrap_or_default().to_string(), setup, sums })
}

/// `a` is a later version than `b` (numeric major.minor.patch).
fn newer(a: &str, b: &str) -> bool {
    let parts = |s: &str| s.split('.').map(|p| p.trim().parse::<u64>().unwrap_or(0)).collect::<Vec<_>>();
    parts(a) > parts(b)
}

/// The hash `sha256sum` printed for `name`.
fn sum_for(sums: &str, name: &str) -> Option<String> {
    sums.lines().find_map(|l| {
        let (hash, file) = l.split_once(char::is_whitespace)?;
        (file.trim().trim_start_matches('*') == name).then(|| hash.to_lowercase())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_by_number() {
        assert!(newer("1.10.0", "1.9.3"));
        assert!(newer("1.2.1", "1.2.0"));
        assert!(!newer("1.2.0", "1.2.0"));
        assert!(!newer("1.1.9", "1.2.0"));
    }

    #[test]
    fn checksums_are_read_from_sha256sum_output() {
        let sums = "abc123  Dotsmith-1.2.1-setup.exe\nDEF456  Dotsmith-1.2.1-legacy-setup.exe\n";
        assert_eq!(sum_for(sums, "Dotsmith-1.2.1-legacy-setup.exe").as_deref(), Some("def456"));
        assert_eq!(sum_for(sums, "Dotsmith-1.2.1-win64.zip"), None);
    }

    #[test]
    fn a_release_picks_this_builds_installer() {
        let v = serde_json::json!({
            "tag_name": "v9.0.0", "html_url": "https://github.com/x/y/releases/tag/v9.0.0",
            "assets": [
                {"name": "Dotsmith-9.0.0-setup.exe", "browser_download_url": "https://a/std", "size": 10},
                {"name": "Dotsmith-9.0.0-legacy-setup.exe", "browser_download_url": "https://a/leg", "size": 20},
                {"name": "SHA256SUMS.txt", "browser_download_url": "https://a/sums", "size": 1}
            ]
        });
        let r = parse_release(&v).unwrap();
        assert_eq!(r.version, "9.0.0");
        assert_eq!(r.setup.unwrap().name, setup_name("9.0.0"));
        assert_eq!(r.sums.as_deref(), Some("https://a/sums"));
    }

    #[test]
    #[ignore = "network: asks GitHub for the latest release"]
    fn the_latest_release_has_this_builds_installer() {
        let v: Value = get(&format!("https://api.github.com/repos/{REPO}/releases/latest")).unwrap().into_json().unwrap();
        let r = parse_release(&v).unwrap();
        assert!(r.setup.is_some() && r.sums.is_some(), "{r:?}");
    }
}

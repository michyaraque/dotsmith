//! Live model information from Hugging Face: the model card, stats, file sizes and the publisher's avatar. Fetched
//! on a background thread and cached in `%LOCALAPPDATA%\Dotsmith\cache\hf` for a day, so the app still shows it offline.
use gpui::RenderImage;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

const TTL: Duration = Duration::from_secs(24 * 3600);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RepoInfo {
    pub repo: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    /// ISO date of the last commit.
    pub last_modified: String,
    pub license: String,
    pub task: String,
    pub gated: bool,
    pub parameters: u64,
    /// The model card without its front matter, or empty when the repo is gated and there is no token.
    pub readme: String,
    /// Bytes per file path, for the paths asked for.
    pub sizes: HashMap<String, u64>,
    pub fetched: u64,
}

#[derive(Clone)]
pub enum State {
    Loading,
    Ready(Arc<RepoInfo>, Option<Arc<RenderImage>>),
    Failed(String),
}

#[derive(Default)]
pub struct Hub {
    repos: Arc<Mutex<HashMap<String, State>>>,
    changed: Arc<AtomicBool>,
}

fn cache_dir() -> PathBuf {
    crate::settings::log_dir().parent().map(|p| p.join("cache").join("hf")).unwrap_or_else(|| std::env::temp_dir().join("dotsmith-hf"))
}

fn cache_name(repo: &str) -> String {
    repo.replace('/', "--")
}

fn now() -> u64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Hub {
    pub fn get(&self, repo: &str) -> Option<State> {
        self.repos.lock().unwrap().get(repo).cloned()
    }

    /// True once after any repo finished loading (the UI repaints).
    pub fn take_changed(&self) -> bool {
        self.changed.swap(false, Ordering::SeqCst)
    }

    /// Start loading `repo` (with the sizes of `paths`) unless it is loaded or loading. A fresh cache is used as is.
    pub fn request(&self, repo: &str, paths: Vec<String>, token: Option<String>) {
        if repo.is_empty() || self.repos.lock().unwrap().contains_key(repo) {
            return;
        }
        self.repos.lock().unwrap().insert(repo.to_string(), State::Loading);
        let (repos, changed, repo) = (self.repos.clone(), self.changed.clone(), repo.to_string());
        std::thread::Builder::new()
            .name("hf".into())
            .spawn(move || {
                let cached = read_cache(&repo);
                let fresh = cached.as_ref().is_some_and(|c| now().saturating_sub(c.fetched) < TTL.as_secs() && paths.iter().all(|p| c.sizes.contains_key(p)));
                let info = match cached {
                    Some(c) if fresh => Ok(c),
                    cached => fetch(&repo, &paths, token.as_deref()).inspect(write_cache).or_else(|e| cached.ok_or(e)),
                };
                let state = match info {
                    Ok(i) => {
                        let avatar = avatar(&i.author);
                        State::Ready(Arc::new(i), avatar)
                    }
                    Err(e) => State::Failed(e),
                };
                repos.lock().unwrap().insert(repo, state);
                changed.store(true, Ordering::SeqCst);
            })
            .ok();
    }

    /// Forget everything loaded, so the next `request` reads Hugging Face again.
    pub fn refresh(&self) {
        self.repos.lock().unwrap().clear();
        let _ = std::fs::remove_dir_all(cache_dir());
    }
}

fn read_cache(repo: &str) -> Option<RepoInfo> {
    let t = std::fs::read_to_string(cache_dir().join(format!("{}.json", cache_name(repo)))).ok()?;
    serde_json::from_str(&t).ok()
}

fn write_cache(info: &RepoInfo) {
    let dir = cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join(format!("{}.json", cache_name(&info.repo))), serde_json::to_string(info).unwrap_or_default());
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().user_agent(concat!("Dotsmith/", env!("CARGO_PKG_VERSION"))).timeout(Duration::from_secs(20)).build()
}

#[allow(clippy::result_large_err, reason = "ureq's own error, matched by the callers for the HTTP status")]
fn get(agent: &ureq::Agent, url: &str, token: Option<&str>) -> Result<ureq::Response, ureq::Error> {
    let r = agent.get(url);
    match token {
        Some(t) => r.set("Authorization", &format!("Bearer {t}")).call(),
        None => r.call(),
    }
}

fn fetch(repo: &str, paths: &[String], token: Option<&str>) -> Result<RepoInfo, String> {
    let agent = agent();
    let api: Value =
        get(&agent, &format!("https://huggingface.co/api/models/{repo}"), token).map_err(|e| e.to_string())?.into_json().map_err(|e| e.to_string())?;
    let mut info = RepoInfo {
        repo: repo.to_string(),
        author: api["author"].as_str().unwrap_or_else(|| repo.split('/').next().unwrap_or("")).to_string(),
        downloads: api["downloads"].as_u64().unwrap_or(0),
        likes: api["likes"].as_u64().unwrap_or(0),
        last_modified: api["lastModified"].as_str().unwrap_or("").to_string(),
        license: api["cardData"]["license_name"].as_str().or_else(|| api["cardData"]["license"].as_str()).unwrap_or("").to_string(),
        task: api["pipeline_tag"].as_str().unwrap_or("").to_string(),
        gated: api["gated"].as_bool().unwrap_or(api["gated"].is_string()),
        parameters: api["safetensors"]["total"].as_u64().unwrap_or(0),
        fetched: now(),
        ..Default::default()
    };
    if let Ok(r) = get(&agent, &format!("https://huggingface.co/{repo}/raw/main/README.md"), token) {
        info.readme = strip_front_matter(&r.into_string().unwrap_or_default());
    }
    let mut dirs: Vec<&str> = paths.iter().map(|p| p.rsplit_once('/').map_or("", |(d, _)| d)).collect();
    dirs.sort();
    dirs.dedup();
    for d in dirs {
        let url = format!("https://huggingface.co/api/models/{repo}/tree/main/{d}");
        let Ok(r) = get(&agent, &url, token) else { continue };
        let Ok(list) = r.into_json::<Vec<Value>>() else { continue };
        for f in list {
            let size = f["lfs"]["size"].as_u64().or_else(|| f["size"].as_u64());
            if let (Some(p), Some(n)) = (f["path"].as_str(), size)
                && paths.iter().any(|q| q == p)
            {
                info.sizes.insert(p.to_string(), n);
            }
        }
    }
    Ok(info)
}

/// The publisher's avatar, from the disk cache or Hugging Face (organisations first, then users).
fn avatar(author: &str) -> Option<Arc<RenderImage>> {
    if author.is_empty() {
        return None;
    }
    let file = cache_dir().join(format!("{}.avatar", cache_name(author)));
    let bytes = std::fs::read(&file).ok().or_else(|| {
        let agent = agent();
        let url = ["organizations", "users"].iter().find_map(|kind| {
            let v: Value = get(&agent, &format!("https://huggingface.co/api/{kind}/{author}/avatar"), None).ok()?.into_json().ok()?;
            v["avatarUrl"].as_str().map(String::from)
        })?;
        let mut buf = vec![];
        std::io::Read::read_to_end(&mut get(&agent, &url, None).ok()?.into_reader(), &mut buf).ok()?;
        let _ = std::fs::create_dir_all(cache_dir());
        let _ = std::fs::write(&file, &buf);
        Some(buf)
    })?;
    let im = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let im = image::imageops::resize(&im, 96, 96, image::imageops::FilterType::Triangle);
    Some(crate::pixels::render(&im, 1))
}

fn strip_front_matter(md: &str) -> String {
    let t = md.trim_start_matches('\u{feff}');
    if let Some(rest) = t.strip_prefix("---")
        && let Some(end) = rest.find("\n---")
    {
        return rest[end + 4..].trim_start().to_string();
    }
    t.to_string()
}

/// One block of a model card, reduced to what the app draws: headings, paragraphs and list items as plain text.
#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    Heading(String),
    Paragraph(String),
    Bullet(String),
}

fn plain(line: &str) -> String {
    let mut s = line.to_string();
    // images, then links keep their text, then html tags and emphasis go
    while let Some(i) = s.find("![") {
        let Some(j) = s[i..].find(')') else { break };
        s.replace_range(i..i + j + 1, "");
    }
    while let Some(i) = s.find('[') {
        let Some(mid) = s[i..].find("](") else { break };
        let Some(end) = s[i + mid..].find(')') else { break };
        let text = s[i + 1..i + mid].to_string();
        s.replace_range(i..i + mid + end + 1, &text);
    }
    while let Some(i) = s.find('<') {
        let Some(j) = s[i..].find('>') else { break };
        s.replace_range(i..i + j + 1, "");
    }
    s.replace("**", "").replace("__", "").replace('`', "").replace("&nbsp;", " ").split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The card as blocks, without images, badges, tables or code.
pub fn blocks(md: &str) -> Vec<Block> {
    let mut out = vec![];
    let mut para = String::new();
    let mut in_code = false;
    let flush = |para: &mut String, out: &mut Vec<Block>| {
        let p = plain(para);
        if p.chars().filter(|c| c.is_alphabetic()).count() > 20 {
            out.push(Block::Paragraph(p));
        }
        para.clear();
    };
    for line in md.lines() {
        let l = line.trim();
        if l.starts_with("```") {
            in_code = !in_code;
            flush(&mut para, &mut out);
            continue;
        }
        if in_code || l.starts_with('|') {
            continue;
        }
        if l.is_empty() {
            flush(&mut para, &mut out);
        } else if let Some(h) = l.strip_prefix('#') {
            flush(&mut para, &mut out);
            let h = plain(h.trim_start_matches('#'));
            if !h.is_empty() {
                out.push(Block::Heading(h));
            }
        } else if let Some(b) = l.strip_prefix("- ").or_else(|| l.strip_prefix("* ")) {
            flush(&mut para, &mut out);
            let b = plain(b);
            if !b.is_empty() {
                out.push(Block::Bullet(b));
            }
        } else {
            para.push(' ');
            para.push_str(l);
        }
    }
    flush(&mut para, &mut out);
    out
}

/// The first paragraph of the card that reads as prose (not a row of links or badges): what a list shows under the name.
pub fn summary(md: &str) -> String {
    blocks(md)
        .into_iter()
        .find_map(|b| match b {
            Block::Paragraph(p) if !p.contains(" | ") && p.split_whitespace().count() >= 12 => Some(p),
            _ => None,
        })
        .unwrap_or_default()
}

/// "3 days ago" style age of an ISO date ("2026-07-24T15:50:25.000Z").
pub fn age(iso: &str) -> String {
    let d: Vec<i64> = iso.get(..10).unwrap_or("").split('-').filter_map(|x| x.parse().ok()).collect();
    if d.len() != 3 {
        return String::new();
    }
    let days_from_civil = |y: i64, m: i64, d: i64| {
        let y = if m <= 2 { y - 1 } else { y };
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
        era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468
    };
    let days = (now() as i64 / 86_400 - days_from_civil(d[0], d[1], d[2])).max(0);
    match days {
        0 => crate::tr!("hoy", "today").into(),
        1 => crate::tr!("ayer", "yesterday").into(),
        2..=59 => crate::tr!(format!("hace {days} días"), format!("{days} days ago")),
        60..=729 => crate::tr!(format!("hace {} meses", days / 30), format!("{} months ago", days / 30)),
        _ => crate::tr!(format!("hace {} años", days / 365), format!("{} years ago", days / 365)),
    }
}

/// 106349 -> "106.3K".
pub fn count(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}K", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cards_become_plain_blocks() {
        let md = "---\nlicense: other\n---\n<p><img src=x></p>\n\n## Introduction\n\nWe are excited to open-source **Qwen-Image**, a [model](https://x) for images and editing.\n\n- **Fast** turbo variant\n\n```\ncode\n```\n| a | b |\n";
        let b = blocks(&strip_front_matter(md));
        assert_eq!(b[0], Block::Heading("Introduction".into()));
        assert_eq!(b[1], Block::Paragraph("We are excited to open-source Qwen-Image, a model for images and editing.".into()));
        assert_eq!(b[2], Block::Bullet("Fast turbo variant".into()));
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn counts_read_short() {
        assert_eq!(count(950), "950");
        assert_eq!(count(106_349), "106.3K");
        assert_eq!(count(2_500_000), "2.5M");
    }

    /// Reads a public repo from Hugging Face: `cargo test --release reads_a_repo -- --ignored`.
    #[test]
    #[ignore]
    fn reads_a_repo() {
        let i = fetch("Qwen/Qwen-Image-2.1", &["processor/tokenizer.json".into()], None).unwrap();
        assert_eq!(i.author, "Qwen");
        assert!(i.downloads > 0 && !summary(&i.readme).is_empty());
        assert!(i.sizes["processor/tokenizer.json"] > 1_000_000);
        assert!(avatar(&i.author).is_some());
    }
}

//! The model catalog as data: what Dotsmith needs to know that Hugging Face can't tell it (which architecture runs a
//! family, which file plays which role, how its output is post-processed). Names, descriptions, sizes, stats and
//! avatars come live from Hugging Face. `models.json` ships inside the binary; a user file (set by the app) is
//! merged on top, objects key by key, anything else replaced.
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::model::Family;

const BUILTIN: &str = include_str!("../models.json");

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct VariantSpec {
    pub name: String,
    pub label: String,
    /// Default sampling steps (0: the model's own).
    #[serde(default)]
    pub steps: usize,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Variants {
    pub base: VariantSpec,
    pub turbo: VariantSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Source {
    pub repo: String,
    pub path: String,
}

impl Source {
    pub fn url(&self) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", self.repo, self.path)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct FileSpec {
    pub role: String,
    /// "turbo" when only that variant needs the file.
    #[serde(default)]
    pub variant: Option<String>,
    /// An optional alternative of the default file with the same role and variant, in another precision ("bf16",
    /// "fp8", ...). Used when the memory policy prefers that precision and the file is on disk; never required.
    #[serde(default)]
    pub precision: Option<String>,
    /// Path inside the models folder.
    pub file: String,
    /// Other places to look: `{hf:Name}/…`, `{hub:owner/repo}/…`, `{env:VAR}/…`.
    #[serde(default)]
    pub also: Vec<String>,
    /// An environment variable that names the file directly and wins over every other location.
    #[serde(default)]
    pub env: Option<String>,
    #[serde(default)]
    pub source: Option<Source>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct FamilySpec {
    pub label: String,
    /// The Hugging Face repo whose model card describes the family.
    pub repo: String,
    pub variants: Variants,
    /// Image pixels per sprite pixel without a LoRA saying otherwise.
    pub scale: u32,
    /// Flat key colour the outputs are painted on ("#RRGGBB"), None = real transparency.
    pub background: Option<String>,
    /// Whether the negative prompt is used (some families burn their output with it under CFG).
    pub negative_prompt: bool,
    /// Whether guidance and Turbo steps always follow the family defaults.
    pub fixed_guidance: bool,
    pub capabilities: Vec<String>,
    /// Module names only this architecture has (`N` = any block number), as a LoRA names them after
    /// `diffusion_model.` (or kohya's `lora_unet_` with `_` for `.`): how a loose LoRA file is matched to its family.
    pub lora_modules: Vec<String>,
    /// "plain": the model reads the description as written (no trigger, layout words or size flags).
    pub prompt: String,
    /// Square sizes the model draws, in sprite pixels; empty = any.
    pub sizes: Vec<u32>,
    /// Studio modes it supports (mode keys); empty = all.
    pub modes: Vec<String>,
    /// Whether "Opaque background" means something for it (a painted background on request).
    #[serde(default = "yes")]
    pub opaque: bool,
    pub files: Vec<FileSpec>,
}

fn yes() -> bool {
    true
}

impl FamilySpec {
    pub fn background_rgb(&self) -> Option<[u8; 3]> {
        parse_hex(self.background.as_deref()?)
    }

    pub fn can(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }
}

pub fn parse_hex(s: &str) -> Option<[u8; 3]> {
    let s = s.trim_start_matches('#');
    if s.len() != 6 {
        return None;
    }
    let v = u32::from_str_radix(s, 16).ok()?;
    Some([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

struct Loaded {
    families: serde_json::Map<String, Value>,
    store: Vec<StoreEntry>,
    user_error: Option<String>,
}

/// A model the catalog lists before Dotsmith can run it: shown in the model manager with its Hugging Face card, never
/// downloadable.
#[derive(Clone, Debug, Deserialize)]
pub struct StoreEntry {
    pub repo: String,
    pub label: String,
}

static USER_FILE: RwLock<Option<PathBuf>> = RwLock::new(None);
static CACHE: RwLock<Option<Arc<Loaded>>> = RwLock::new(None);

/// Merge a user catalog on top of the built-in one (None = built-in only), and re-read it.
pub fn set_user_file(path: Option<PathBuf>) {
    *USER_FILE.write().unwrap() = path;
    *CACHE.write().unwrap() = None;
}

fn merge(base: &mut Value, over: Value) {
    match (base, over) {
        (Value::Object(b), Value::Object(o)) => {
            for (k, v) in o {
                merge(b.entry(k).or_insert(Value::Null), v);
            }
        }
        (b, o) => *b = o,
    }
}

fn load() -> Arc<Loaded> {
    if let Some(l) = CACHE.read().unwrap().as_ref() {
        return l.clone();
    }
    let mut all: Value = serde_json::from_str(BUILTIN).expect("models.json is valid");
    let mut user_error = None;
    if let Some(p) = USER_FILE.read().unwrap().clone().filter(|p| p.is_file()) {
        match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string())) {
            Ok(v) => merge(&mut all, v),
            Err(e) => user_error = Some(e),
        }
    }
    let families = all["families"].as_object().cloned().unwrap_or_default();
    let store = serde_json::from_value(all["store"].clone()).unwrap_or_default();
    let l = Arc::new(Loaded { families, store, user_error });
    *CACHE.write().unwrap() = Some(l.clone());
    l
}

pub fn family(f: Family) -> FamilySpec {
    load().families.get(f.id()).cloned().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
}

/// Models listed as coming soon.
pub fn store() -> Vec<StoreEntry> {
    load().store.clone()
}

/// Why the user catalog was ignored, if it was.
pub fn user_error() -> Option<String> {
    load().user_error.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_architecture_is_described() {
        for f in Family::ALL {
            let s = family(f);
            assert!(!s.label.is_empty() && s.scale > 0 && !s.files.is_empty(), "{}", f.id());
        }
    }

    #[test]
    fn a_user_catalog_overrides_fields_and_keeps_the_rest() {
        let mut base: Value = serde_json::json!({"families": {"a": {"label": "A", "scale": 16, "files": [1, 2]}}});
        merge(&mut base, serde_json::json!({"families": {"a": {"scale": 8, "files": [3]}}}));
        assert_eq!(base["families"]["a"], serde_json::json!({"label": "A", "scale": 8, "files": [3]}));
    }
}

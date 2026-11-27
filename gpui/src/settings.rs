//! User settings in `%APPDATA%\Dotsmith\settings.json`. A missing or unreadable file means defaults.
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::i18n::{self, Lang};
use dotsmith_engine::model::Family;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct WindowState {
    pub width: f32,
    pub height: f32,
    pub maximized: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// None = the Windows display language.
    pub language: Option<String>,
    /// "dark" or "light"; None = follow Windows.
    pub appearance: Option<String>,
    pub models_dir: Option<PathBuf>,
    pub hf_dirs: Vec<PathBuf>,
    /// None = `%LOCALAPPDATA%\Dotsmith\loras`.
    pub lora_dir: Option<PathBuf>,
    /// None = Pictures\Dotsmith.
    pub output_dir: Option<PathBuf>,
    /// None = automatic.
    pub low_vram: Option<bool>,
    /// Where model weights live (GPU or streamed from RAM), VRAM limit, RAM mode, images per pass.
    pub memory: dotsmith_engine::core::memory::MemoryPolicy,
    /// For gated Hugging Face repos. HF_TOKEN is used when this is empty.
    pub hf_token: Option<String>,
    pub window: Option<WindowState>,
    pub last_mode: Option<String>,
    /// LoRA strength (None = 1.0).
    pub lora_scale: Option<f32>,
    /// Look for new versions at start and every 12 hours (None = on).
    pub check_updates: Option<bool>,
    /// What the sprite modes wrap the description in (None = jobs::TEMPLATE).
    pub prompt_template: Option<String>,
}

pub fn config_dir() -> Option<PathBuf> {
    std::env::var_os("APPDATA")
        .map(|a| PathBuf::from(a).join("Dotsmith"))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config/dotsmith")))
}

pub fn log_dir() -> PathBuf {
    std::env::var_os("LOCALAPPDATA").map(|a| PathBuf::from(a).join("Dotsmith").join("logs")).unwrap_or_else(|| std::env::temp_dir().join("dotsmith-logs"))
}

fn path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("settings.json"))
}

impl Settings {
    pub fn load() -> Settings {
        path().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(p) = path() else { return Ok(()) };
        std::fs::create_dir_all(p.parent().unwrap())?;
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self).unwrap_or_default())?;
        std::fs::rename(tmp, p)
    }

    pub fn hf_token(&self) -> Option<String> {
        self.hf_token.clone().filter(|t| !t.trim().is_empty()).or_else(|| std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()))
    }

    pub fn light(&self) -> bool {
        match self.appearance.as_deref() {
            Some("light") => true,
            Some("dark") => false,
            _ => crate::theme::system_prefers_light(),
        }
    }

    pub fn lang(&self) -> Lang {
        self.language.as_deref().and_then(Lang::from_code).unwrap_or_else(i18n::system)
    }

    /// Takes effect on the next model load.
    pub fn apply_to_engine(&self) {
        dotsmith_engine::manifest::set_user_file(config_dir().map(|d| d.join("models.json")));
        dotsmith_engine::locations::set(dotsmith_engine::locations::Locations { models_dir: self.models_dir.clone(), hf_dirs: self.hf_dirs.clone() });
        dotsmith_engine::core::set_low_vram(self.low_vram);
        dotsmith_engine::core::memory::set_policy(self.memory.clone());
    }

    pub fn lora_library(&self) -> Option<PathBuf> {
        match &self.lora_dir {
            Some(d) => Some(library_dir(d)),
            None => default_library(),
        }
    }

    pub fn fallback_output_dir() -> PathBuf {
        pictures_dir().join("Dotsmith")
    }
}

/// A path as shown in the interface: the user's profile folders as variables, never the account name.
pub fn display_path(p: &Path) -> String {
    let text = p.display().to_string();
    for var in ["LOCALAPPDATA", "APPDATA", "USERPROFILE"] {
        if let Some(v) = std::env::var_os(var).map(|v| v.to_string_lossy().to_string()).filter(|v| !v.is_empty())
            && text.to_lowercase().starts_with(&v.to_lowercase())
        {
            return format!("%{var}%{}", &text[v.len()..]);
        }
    }
    text
}

/// The library a picked folder stands for: the folder itself, or its parent when a training family folder was picked.
pub fn library_dir(d: &Path) -> PathBuf {
    let is_family = d.file_name().and_then(|n| n.to_str()).and_then(Family::from_id).is_some();
    match d.parent().filter(|_| is_family) {
        Some(parent) => parent.to_path_buf(),
        None => d.to_path_buf(),
    }
}

/// `%LOCALAPPDATA%\Dotsmith\loras`, created so the user can open it and drop LoRAs in.
pub fn default_library() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("Dotsmith").join("loras");
    let _ = std::fs::create_dir_all(&dir);
    Some(dir)
}

pub fn pictures_dir() -> PathBuf {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct Guid(u32, u16, u16, [u8; 8]);
        const FOLDERID_PICTURES: Guid = Guid(0x33E28130, 0x4E1E, 0x4676, [0x83, 0x5A, 0x98, 0x39, 0x5C, 0x3B, 0xC3, 0xBB]);
        unsafe extern "system" {
            fn SHGetKnownFolderPath(id: *const Guid, flags: u32, token: isize, out: *mut *mut u16) -> i32;
            fn CoTaskMemFree(p: *mut std::ffi::c_void);
        }
        let mut out: *mut u16 = std::ptr::null_mut();
        if unsafe { SHGetKnownFolderPath(&FOLDERID_PICTURES, 0, 0, &mut out) } == 0 && !out.is_null() {
            let len = (0..).take_while(|&i| unsafe { *out.add(i) } != 0).count();
            let s = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(out, len) });
            unsafe { CoTaskMemFree(out.cast()) };
            return PathBuf::from(s);
        }
    }
    std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(|h| PathBuf::from(h).join("Pictures")).unwrap_or_else(std::env::temp_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_tolerates_unknown_or_missing_fields() {
        let s = Settings { language: Some("en".into()), models_dir: Some("C:/m".into()), low_vram: Some(true), ..Default::default() };
        let t = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Settings>(&t).unwrap(), s);
        let old: Settings = serde_json::from_str(r#"{"language":"es","something_new":1}"#).unwrap();
        assert_eq!(old.language.as_deref(), Some("es"));
        assert!(old.models_dir.is_none());
        assert_eq!(old.memory, dotsmith_engine::core::memory::MemoryPolicy::default());
        let partial: Settings = serde_json::from_str(r#"{"memory":{"placement":"stream","vram_limit_mb":8000}}"#).unwrap();
        assert_eq!(partial.memory.placement, dotsmith_engine::core::memory::Placement::Stream);
        assert_eq!(partial.memory.vram_limit_mb, Some(8000));
        assert_eq!(partial.memory.batch_max, 4);
    }

    #[test]
    fn shown_paths_hide_the_account_folder() {
        let Some(local) = std::env::var_os("LOCALAPPDATA") else { return };
        let p = PathBuf::from(local).join("Dotsmith").join("models");
        assert_eq!(display_path(&p), format!("%LOCALAPPDATA%{}Dotsmith{}models", std::path::MAIN_SEPARATOR, std::path::MAIN_SEPARATOR));
        assert_eq!(display_path(Path::new("D:/Models")), "D:/Models");
    }

    #[test]
    fn library_is_the_picked_folder_or_the_parent_of_a_family_folder() {
        let lib = Path::new("E:/LoRAs");
        assert_eq!(library_dir(lib), lib);
        assert_eq!(library_dir(&lib.join(Family::ALL[0].id())), lib);
    }
}

//! What an install touches outside its folder: shortcuts, the entry in Windows' installed apps, and the install of
//! the versions packaged with Inno Setup (1.0 and 1.1), which this one replaces.
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use winreg::RegKey;
use winreg::enums::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_READ};

const UNINSTALL: &str = r"Software\Microsoft\Windows\CurrentVersion\Uninstall";
const KEY: &str = "Dotsmith";
/// Inno Setup's key for Dotsmith 1.0 and 1.1 (its AppId plus "_is1")
const INNO_KEY: &str = "{6F0C2B9E-5A41-4C7E-9F0B-3E1D2A7C8B55}_is1";
pub const EXE: &str = "dotsmith.exe";
pub const UNINSTALLER: &str = "uninstall.exe";

fn env_dir(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(PathBuf::from).filter(|p| p.is_absolute())
}

/// `%LOCALAPPDATA%\Programs\Dotsmith`: per user, no admin rights (where VS Code or Discord go).
pub fn default_dir() -> PathBuf {
    env_dir("LOCALAPPDATA").unwrap_or_else(std::env::temp_dir).join("Programs").join("Dotsmith")
}

/// Downloaded models in their default place (an uninstall may remove them).
pub fn models_dir() -> Option<PathBuf> {
    env_dir("LOCALAPPDATA").map(|d| d.join("Dotsmith").join("models"))
}

/// Settings, the last model and logs (an uninstall may remove them); LoRAs and images are never touched.
pub fn settings_dirs() -> Vec<PathBuf> {
    [env_dir("APPDATA").map(|d| d.join("Dotsmith")), env_dir("LOCALAPPDATA").map(|d| d.join("Dotsmith").join("logs"))].into_iter().flatten().collect()
}

pub fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => dir_size(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

/// Dotsmith as Windows lists it in Settings > Apps.
#[derive(Clone, Debug)]
pub struct Installed {
    pub version: String,
    pub dir: PathBuf,
}

pub fn installed() -> Option<Installed> {
    let k = RegKey::predef(HKEY_CURRENT_USER).open_subkey(format!(r"{UNINSTALL}\{KEY}")).ok()?;
    let dir: String = k.get_value("InstallLocation").ok()?;
    Some(Installed { version: k.get_value("DisplayVersion").unwrap_or_default(), dir: PathBuf::from(dir) })
}

pub fn register(dir: &Path, version: &str, bytes: u64) -> Result<()> {
    let (k, _) = RegKey::predef(HKEY_CURRENT_USER).create_subkey(format!(r"{UNINSTALL}\{KEY}"))?;
    let uninstaller = dir.join(UNINSTALLER);
    k.set_value("DisplayName", &"Dotsmith")?;
    k.set_value("DisplayVersion", &version)?;
    k.set_value("Publisher", &"Michael Araque")?;
    k.set_value("URLInfoAbout", &"https://github.com/michyaraque/dotsmith")?;
    k.set_value("DisplayIcon", &dir.join(EXE).to_string_lossy().to_string())?;
    k.set_value("InstallLocation", &dir.to_string_lossy().to_string())?;
    k.set_value("UninstallString", &format!("\"{}\" --uninstall", uninstaller.display()))?;
    k.set_value("QuietUninstallString", &format!("\"{}\" --uninstall --silent", uninstaller.display()))?;
    k.set_value("EstimatedSize", &((bytes >> 10) as u32))?;
    k.set_value("NoModify", &1u32)?;
    k.set_value("NoRepair", &1u32)?;
    Ok(())
}

pub fn unregister() {
    let _ = RegKey::predef(HKEY_CURRENT_USER).open_subkey(UNINSTALL).and_then(|k| k.delete_subkey_all(KEY));
}

/// Dotsmith 1.0 or 1.1, installed by Inno Setup (for all users or this one).
pub struct Legacy {
    pub version: String,
    uninstaller: PathBuf,
    all_users: bool,
}

pub fn legacy() -> Option<Legacy> {
    [(HKEY_LOCAL_MACHINE, true), (HKEY_CURRENT_USER, false)].into_iter().find_map(|(root, all_users)| {
        let k = RegKey::predef(root).open_subkey_with_flags(format!(r"{UNINSTALL}\{INNO_KEY}"), KEY_READ).ok()?;
        let cmd: String = k.get_value("UninstallString").ok()?;
        let version = k.get_value("DisplayVersion").unwrap_or_else(|_| "1.0".to_string());
        Some(Legacy { version, uninstaller: PathBuf::from(cmd.trim_matches('"')), all_users })
    })
}

impl Legacy {
    /// Run Inno's uninstaller quietly and wait until its entry is gone. An install for all users asks Windows for
    /// admin rights once (it lives in Program Files).
    pub fn remove(&self) -> Result<()> {
        if !self.uninstaller.is_file() {
            return Ok(()); // a stale entry: nothing left to run
        }
        shell_run(&self.uninstaller, "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART")?;
        // Inno's uninstaller hands over to a temporary copy and exits at once: its registry entry is the signal
        let root = if self.all_users { HKEY_LOCAL_MACHINE } else { HKEY_CURRENT_USER };
        let start = Instant::now();
        while RegKey::predef(root).open_subkey_with_flags(format!(r"{UNINSTALL}\{INNO_KEY}"), KEY_READ).is_ok() {
            if start.elapsed() > Duration::from_secs(180) {
                anyhow::bail!("the previous version did not finish uninstalling");
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        Ok(())
    }
}

/// Start `exe` through the shell (which shows the admin prompt when the program needs one) and wait for it.
fn shell_run(exe: &Path, args: &str) -> Result<()> {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};
    use windows::Win32::UI::Shell::{SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW, ShellExecuteExW};
    use windows::Win32::UI::WindowsAndMessaging::SW_HIDE;
    use windows::core::{HSTRING, PCWSTR};
    let (file, params) = (HSTRING::from(exe.as_os_str()), HSTRING::from(args));
    let mut info = SHELLEXECUTEINFOW {
        cbSize: std::mem::size_of::<SHELLEXECUTEINFOW>() as u32,
        fMask: SEE_MASK_NOCLOSEPROCESS,
        lpFile: PCWSTR(file.as_ptr()),
        lpParameters: PCWSTR(params.as_ptr()),
        nShow: SW_HIDE.0,
        ..Default::default()
    };
    unsafe {
        ShellExecuteExW(&mut info).context("could not start the previous version's uninstaller")?;
        if !info.hProcess.is_invalid() {
            WaitForSingleObject(info.hProcess, INFINITE);
            let _ = CloseHandle(info.hProcess);
        }
    }
    Ok(())
}

/// Start menu and desktop shortcuts to `dotsmith.exe`.
pub fn start_menu_link() -> Option<PathBuf> {
    known_folder(&windows::Win32::UI::Shell::FOLDERID_Programs).map(|d| d.join("Dotsmith.lnk"))
}

pub fn desktop_link() -> Option<PathBuf> {
    known_folder(&windows::Win32::UI::Shell::FOLDERID_Desktop).map(|d| d.join("Dotsmith.lnk"))
}

fn known_folder(id: &windows::core::GUID) -> Option<PathBuf> {
    use windows::Win32::System::Com::CoTaskMemFree;
    use windows::Win32::UI::Shell::{KNOWN_FOLDER_FLAG, SHGetKnownFolderPath};
    unsafe {
        let p = SHGetKnownFolderPath(id, KNOWN_FOLDER_FLAG(0), None).ok()?;
        let s = p.to_string().ok();
        CoTaskMemFree(Some(p.0 as _));
        s.map(PathBuf::from)
    }
}

pub fn create_link(link: &Path, target: &Path) -> Result<()> {
    use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx, IPersistFile};
    use windows::Win32::UI::Shell::{IShellLinkW, ShellLink};
    use windows::core::{HSTRING, Interface};
    if let Some(d) = link.parent() {
        std::fs::create_dir_all(d)?;
    }
    unsafe {
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
        let l: IShellLinkW = CoCreateInstance(&ShellLink, None, CLSCTX_INPROC_SERVER)?;
        l.SetPath(&HSTRING::from(target.as_os_str()))?;
        if let Some(d) = target.parent() {
            l.SetWorkingDirectory(&HSTRING::from(d.as_os_str()))?;
        }
        l.SetDescription(&HSTRING::from("Pixel art on your own GPU"))?;
        l.SetIconLocation(&HSTRING::from(target.as_os_str()), 0)?;
        l.cast::<IPersistFile>()?.Save(&HSTRING::from(link.as_os_str()), true)?;
    }
    Ok(())
}

/// Wait until process `pid` has exited (or `limit` passes); an unknown process counts as exited.
pub fn wait_exit(pid: u32, limit: Duration) {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject};
    unsafe {
        if let Ok(h) = OpenProcess(PROCESS_SYNCHRONIZE, false, pid) {
            WaitForSingleObject(h, limit.as_millis() as u32);
            let _ = CloseHandle(h);
        }
    }
}

/// Whether a running program holds `exe` (Windows does not let anyone write to a running program's file).
pub fn in_use(exe: &Path) -> bool {
    exe.is_file() && std::fs::OpenOptions::new().write(true).open(exe).is_err()
}

/// Start `exe` on its own (it outlives this program).
pub fn launch(exe: &Path) {
    let mut c = std::process::Command::new(exe);
    if let Some(d) = exe.parent() {
        c.current_dir(d);
    }
    let _ = c.spawn();
}

/// Delete `path` once this program has exited (a running program cannot delete its own file).
pub fn delete_after_exit(path: &Path) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = std::process::Command::new("cmd")
        .raw_arg(format!("/C ping 127.0.0.1 -n 3 > nul & del /f /q \"{}\"", path.display()))
        .creation_flags(CREATE_NO_WINDOW)
        .spawn();
}

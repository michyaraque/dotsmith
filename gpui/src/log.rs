//! `%LOCALAPPDATA%\Dotsmith\logs\dotsmith.log`, rotated at 2 MB; panics are written there with their backtrace.
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

static FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);

pub fn path() -> PathBuf {
    crate::settings::log_dir().join("dotsmith.log")
}

pub fn init() {
    let p = path();
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    if p.metadata().is_ok_and(|m| m.len() > 2 * 1024 * 1024) {
        let _ = std::fs::rename(&p, p.with_file_name("dotsmith.old.log"));
    }
    *FILE.lock().unwrap() = std::fs::OpenOptions::new().create(true).append(true).open(&p).ok();
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current().name().unwrap_or("?").to_string();
        write("PANIC", &format!("thread '{thread}': {info}\n{}", std::backtrace::Backtrace::force_capture()));
        prev(info);
    }));
    info(&format!("Dotsmith {} starting", env!("CARGO_PKG_VERSION")));
}

fn write(level: &str, msg: &str) {
    if let Some(f) = FILE.lock().ok().as_mut().and_then(|g| g.as_mut()) {
        let _ = writeln!(f, "{} {level:5} {msg}", crate::jobs::chrono_stamp());
        let _ = f.flush();
    }
}

pub fn info(msg: &str) {
    write("INFO", msg);
}

pub fn error(msg: &str) {
    write("ERROR", msg);
}

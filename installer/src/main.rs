#![windows_subsystem = "windows"]
//! Dotsmith Setup: installs Dotsmith for the current user (no admin rights), updates it in place, and is also its
//! uninstaller. The app's files ride behind the program (payload.rs).
//!
//! `dotsmith-setup`                                   install with the window
//! `dotsmith-setup --silent [--dir <folder>] [--desktop]`   install with no window
//! `uninstall.exe --uninstall [--silent]`             uninstall (what Windows' Apps settings run)
//! `dotsmith-setup --pack <folder> <setup.exe>`       packaging: this program + the files of <folder>
#[path = "../../gpui/src/i18n.rs"]
#[allow(dead_code, reason = "shared with the app, which uses all of it")]
mod i18n;
mod payload;
mod system;
mod tasks;
#[path = "../../gpui/src/theme.rs"]
#[allow(dead_code, reason = "shared with the app, which uses all of it")]
mod theme;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;

pub const VERSION: &str = env!("DOTSMITH_VERSION");

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
    i18n::set(i18n::system());
    theme::set_light(theme::system_prefers_light());

    if let Some(i) = args.iter().position(|a| a == "--pack") {
        let (Some(src), Some(out)) = (args.get(i + 1), args.get(i + 2)) else { exit(Err("usage: --pack <folder> <setup.exe>".into())) };
        exit(payload::pack(src.as_ref(), out.as_ref()).map_err(|e| format!("{e:#}")));
    }
    let silent = flag("--silent");

    if flag("--uninstall") {
        let me = std::env::current_exe().unwrap_or_default();
        let Some(dir) = value("--dir").map(PathBuf::from).or_else(|| system::installed().map(|i| i.dir)).or_else(|| me.parent().map(PathBuf::from)) else {
            exit(Err("no install found".into()))
        };
        // the uninstaller is one of the files it removes: run from a copy outside the folder
        if me.starts_with(&dir) {
            let copy = std::env::temp_dir().join(format!("dotsmith-uninstall-{}.exe", std::process::id()));
            let mut again = vec!["--uninstall".to_string(), "--dir".into(), dir.to_string_lossy().into()];
            if silent {
                again.push("--silent".into());
            }
            let r = std::fs::copy(&me, &copy).and_then(|_| std::process::Command::new(&copy).args(again).spawn()).map(|_| ()).map_err(|e| e.to_string());
            exit(r);
        }
        if silent {
            let job = tasks::Uninstall { dir, models: false, settings: false };
            let r = tasks::uninstall(&job, &tasks::Shared::default()).map_err(|e| format!("{e:#}"));
            system::delete_after_exit(&me);
            exit(r);
        }
        ui::run(ui::Mode::Uninstall(dir));
        system::delete_after_exit(&me);
        return;
    }

    let payload = payload::Payload::of_this_program().ok().flatten().map(Arc::new);
    if silent {
        let Some(p) = payload else { exit(Err("this setup has no files".into())) };
        let dir = value("--dir").map(PathBuf::from).or_else(|| system::installed().map(|i| i.dir)).unwrap_or_else(system::default_dir);
        let job = tasks::Install { dir, desktop: flag("--desktop"), version: VERSION.into() };
        exit(tasks::install(&job, &p, &tasks::Shared::default()).map_err(|e| format!("{e:#}")));
    }
    ui::run(ui::Mode::Install(payload));
}

/// End with exit code 0, or 1 after writing the error to stderr (seen when run from a console or a script).
fn exit(r: Result<(), String>) -> ! {
    match r {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1)
        }
    }
}

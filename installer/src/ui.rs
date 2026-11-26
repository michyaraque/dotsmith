//! The setup window, in Dotsmith's own look: a pixel-art panel with the logo on the left, one step per screen on the
//! right (choose > install > done), Enter for the main button and Esc to leave.
use gpui::{
    App, AppContext, Application, AssetSource, Bounds, Context, Div, FocusHandle, FontWeight, KeyBinding, MouseButton, MouseDownEvent, PathPromptOptions,
    RenderImage, SharedString, Stateful, Svg, TitlebarOptions, Window, WindowBounds, WindowControlArea, WindowOptions, actions, div, img, prelude::*, px,
    relative, rgb, size, svg,
};
use image::RgbaImage;
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::payload::Payload;
use crate::tasks::{self, Shared, Step};
use crate::theme::*;
use crate::{VERSION, system, tr};

actions!(setup, [Primary, Back]);

const W: f32 = 780.;
const H: f32 = 500.;
const HERO_W: u32 = 268;
const TITLE_H: f32 = 40.;

pub enum Mode {
    Install(Option<Arc<Payload>>),
    Uninstall(PathBuf),
}

#[derive(Clone, Copy, PartialEq)]
enum Action {
    Install,
    Update,
    Reinstall,
}

#[derive(Clone, PartialEq)]
enum Screen {
    Choose,
    Working,
    Done,
    Failed(String),
    License,
}

pub struct Setup {
    focus: FocusHandle,
    mode: Mode,
    screen: Screen,
    dir: PathBuf,
    /// the version already installed: by this installer (updated in place) or by Inno Setup (`legacy`, 1.0 and 1.1)
    previous: Option<String>,
    legacy: bool,
    desktop: bool,
    launch: bool,
    models: bool,
    settings: bool,
    models_size: u64,
    progress: Shared,
    hero: Arc<RenderImage>,
    logo: Arc<RenderImage>,
}

pub fn run(mode: Mode) {
    Application::new().with_assets(Icons).run(move |cx: &mut App| {
        cx.bind_keys([KeyBinding::new("enter", Primary, None), KeyBinding::new("escape", Back, None)]);
        let bounds = Bounds::centered(None, size(px(W), px(H)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions { title: Some(title(&mode).into()), appears_transparent: true, ..Default::default() }),
                is_resizable: false,
                ..Default::default()
            },
            |window, cx| {
                cx.new(|cx| {
                    let s = Setup::new(mode, cx);
                    window.focus(&s.focus);
                    s
                })
            },
        )
        .unwrap();
        cx.on_window_closed(|cx| cx.quit()).detach();
        cx.activate(true);
    });
}

fn title(mode: &Mode) -> &'static str {
    match mode {
        Mode::Install(_) => tr!("Instalar Dotsmith", "Dotsmith Setup"),
        Mode::Uninstall(_) => tr!("Desinstalar Dotsmith", "Uninstall Dotsmith"),
    }
}

impl Setup {
    fn new(mode: Mode, cx: &mut Context<Self>) -> Self {
        let installed = system::installed();
        let old = if matches!(mode, Mode::Install(_)) && installed.is_none() { system::legacy().map(|l| l.version) } else { None };
        let legacy = old.is_some();
        let (dir, previous) = match (&mode, installed) {
            (Mode::Uninstall(d), i) => (d.clone(), i.map(|i| i.version)),
            (Mode::Install(_), Some(i)) => (i.dir, Some(i.version)),
            (Mode::Install(_), None) => (system::default_dir(), old),
        };
        let screen = match &mode {
            Mode::Install(None) => Screen::Failed(
                tr!(
                    "Este instalador no lleva los archivos de Dotsmith. Descárgalo de nuevo.",
                    "This installer has no Dotsmith files in it. Download it again."
                )
                .into(),
            ),
            _ => Screen::Choose,
        };
        let models_size = match &mode {
            Mode::Uninstall(_) => system::models_dir().map(|d| system::dir_size(&d)).unwrap_or(0),
            Mode::Install(_) => 0,
        };
        // the worker thread reports through `progress`; redraw while it runs
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(Duration::from_millis(50)).await;
                if this.update(cx, |s, cx| s.poll(cx)).is_err() {
                    break;
                }
            }
        })
        .detach();
        Self {
            focus: cx.focus_handle(),
            desktop: system::desktop_link().is_none_or(|l| l.exists() || previous.is_none()),
            mode,
            screen,
            dir,
            previous,
            legacy,
            launch: true,
            models: false,
            settings: false,
            models_size,
            progress: Shared::default(),
            hero: hero(),
            logo: render(&logo(), 1),
        }
    }

    fn action(&self) -> Action {
        match &self.previous {
            Some(v) if !self.legacy && v == VERSION => Action::Reinstall,
            Some(_) => Action::Update,
            _ => Action::Install,
        }
    }

    fn installing(&self) -> bool {
        matches!(self.mode, Mode::Install(_))
    }

    fn poll(&mut self, cx: &mut Context<Self>) {
        if self.screen != Screen::Working {
            return;
        }
        if let Some(r) = self.progress.lock().unwrap().result.take() {
            self.screen = match r {
                Ok(()) => Screen::Done,
                Err(e) => Screen::Failed(e),
            };
        }
        cx.notify();
    }

    fn start(&mut self, cx: &mut Context<Self>) {
        self.progress = Shared::default();
        self.screen = Screen::Working;
        match &self.mode {
            Mode::Install(Some(payload)) => {
                let (payload, job) = (payload.clone(), tasks::Install { dir: self.dir.clone(), desktop: self.desktop, version: VERSION.into() });
                tasks::spawn(self.progress.clone(), move |p| tasks::install(&job, &payload, p));
            }
            Mode::Install(None) => {}
            Mode::Uninstall(dir) => {
                let job = tasks::Uninstall { dir: dir.clone(), models: self.models, settings: self.settings };
                tasks::spawn(self.progress.clone(), move |p| tasks::uninstall(&job, p));
            }
        }
        cx.notify();
    }

    fn primary(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match self.screen {
            Screen::Choose | Screen::Failed(_) if !matches!(self.mode, Mode::Install(None)) => self.start(cx),
            Screen::License => {
                self.screen = Screen::Choose;
                cx.notify();
            }
            Screen::Done => self.finish(window),
            _ => {}
        }
    }

    fn back(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match self.screen {
            Screen::Working => {}
            Screen::License => {
                self.screen = Screen::Choose;
                cx.notify();
            }
            Screen::Done => self.finish(window),
            _ => window.remove_window(),
        }
    }

    /// Closing the window ends the program (gpui's quit does not close it from a click on Windows).
    fn finish(&mut self, window: &mut Window) {
        if self.installing() && self.launch {
            system::launch(&self.dir.join(system::EXE));
        }
        window.remove_window();
    }

    fn pick_dir(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some(tr!("Elegir carpeta", "Choose folder").into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                // a folder of its own, so the uninstall never has more than Dotsmith's files to look at
                let p = if p.file_name().is_some_and(|n| n.eq_ignore_ascii_case("Dotsmith")) { p } else { p.join("Dotsmith") };
                let _ = this.update(cx, |s, cx| {
                    s.dir = p;
                    cx.notify();
                });
            }
        })
        .detach();
    }

    // ------------------------------------------------------------------ screens

    fn choose(&self, cx: &mut Context<Self>) -> Div {
        if !self.installing() {
            let v = self.previous.clone().map(|v| format!(" {v}")).unwrap_or_default();
            return div()
                .flex()
                .flex_col()
                .gap(px(14.))
                .child(heading(tr!(format!("Desinstalar Dotsmith{v}"), format!("Uninstall Dotsmith{v}")), self.dir.display().to_string()))
                .child(note(tr!("Se quitará Dotsmith de este equipo.", "Dotsmith will be removed from this computer.")))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(4.))
                        .when(self.models_size > 0, |d| {
                            d.child(
                                toggle(
                                    "models",
                                    &tr!(
                                        format!("Eliminar también los modelos descargados ({})", bytes(self.models_size)),
                                        format!("Also remove the downloaded models ({})", bytes(self.models_size))
                                    ),
                                    self.models,
                                )
                                .on_click(cx.listener(|s, _, _, cx| {
                                    s.models = !s.models;
                                    cx.notify();
                                })),
                            )
                        })
                        .child(toggle("settings", tr!("Eliminar también la configuración", "Also remove the settings"), self.settings).on_click(cx.listener(
                            |s, _, _, cx| {
                                s.settings = !s.settings;
                                cx.notify();
                            },
                        ))),
                );
        }
        let raw = match &self.mode {
            Mode::Install(Some(p)) => p.raw + p.stub,
            _ => 0,
        };
        let updating = self.previous.is_some() && !self.legacy;
        let title = match self.action() {
            Action::Install => tr!("Instalar Dotsmith", "Install Dotsmith"),
            Action::Update => tr!("Actualizar Dotsmith", "Update Dotsmith"),
            Action::Reinstall => tr!("Reinstalar Dotsmith", "Reinstall Dotsmith"),
        };
        let what = tr!(format!("Versión {VERSION}"), format!("Version {VERSION}"));
        div()
            .flex()
            .flex_col()
            .gap(px(14.))
            .child(heading(title, what))
            .when_some(self.previous.clone(), |d, v| {
                let text = if self.action() == Action::Reinstall {
                    tr!(format!("Dotsmith {v} ya está instalado en este equipo."), format!("Dotsmith {v} is already installed on this computer."))
                } else {
                    tr!(format!("Se actualizará Dotsmith {v} a la versión {VERSION}."), format!("Dotsmith {v} will be updated to version {VERSION}."))
                };
                d.child(note(text))
            })
            .child(
                div().flex().flex_col().gap(px(6.)).child(label(tr!("Carpeta de destino", "Destination folder"))).child(
                    div()
                        .flex()
                        .gap(px(8.))
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .h(px(32.))
                                .px(px(10.))
                                .flex()
                                .items_center()
                                .rounded(px(7.))
                                .bg(rgb(INPUT_BG()))
                                .border_1()
                                .border_color(rgb(BORDER_SOFT()))
                                .text_size(px(12.))
                                .font_family(MONO)
                                .text_color(rgb(MUTED()))
                                .overflow_hidden()
                                .child(div().truncate().child(self.dir.display().to_string())),
                        )
                        .when(!updating, |d| {
                            d.child(button("change", tr!("Cambiar…", "Change…"), Some("folder-open")).on_click(cx.listener(|s, _, _, cx| s.pick_dir(cx))))
                        }),
                ),
            )
            .child(toggle("desktop", tr!("Crear un acceso directo en el escritorio", "Create a desktop shortcut"), self.desktop).on_click(cx.listener(
                |s, _, _, cx| {
                    s.desktop = !s.desktop;
                    cx.notify();
                },
            )))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap(px(4.))
                    .text_size(px(11.5))
                    .text_color(rgb(MUTED_2()))
                    .child(tr!(
                        format!("Espacio necesario: {}. Al instalar aceptas la", bytes(raw)),
                        format!("Space required: {}. By installing you accept the", bytes(raw))
                    ))
                    .child(div().id("license").text_color(rgb(TEXT())).underline().cursor_pointer().child(tr!("licencia.", "license.")).on_click(cx.listener(
                        |s, _, _, cx| {
                            s.screen = Screen::License;
                            cx.notify();
                        },
                    ))),
            )
    }

    fn working(&self) -> Div {
        let p = self.progress.lock().unwrap();
        let v = self.previous.clone().unwrap_or_default();
        let old = tr!(format!("Quitar la versión {v}"), format!("Remove version {v}"));
        let steps: Vec<(Step, &str)> = if self.installing() {
            [
                self.legacy.then_some((Step::Previous, old.as_str())),
                Some((Step::Files, tr!("Copiar archivos", "Copy files"))),
                Some((Step::Shortcuts, tr!("Crear accesos directos", "Create shortcuts"))),
                Some((Step::Register, tr!("Registrar la aplicación", "Register the application"))),
            ]
            .into_iter()
            .flatten()
            .collect()
        } else {
            vec![
                (Step::Files, tr!("Quitar archivos", "Remove files")),
                (Step::Shortcuts, tr!("Quitar accesos directos", "Remove shortcuts")),
                (Step::Register, tr!("Quitar el registro de la aplicación", "Unregister the application")),
            ]
        };
        let current = steps.iter().position(|s| Some(s.0) == p.step).unwrap_or(0);
        let frac = if p.total > 0 { (p.done as f32 / p.total as f32).clamp(0., 1.) } else { 0. };
        let show_bar = self.installing() && p.step == Some(Step::Files);
        div()
            .flex()
            .flex_col()
            .gap(px(18.))
            .child(heading(
                if self.installing() {
                    tr!("Instalando Dotsmith…", "Installing Dotsmith…")
                } else {
                    tr!("Desinstalando Dotsmith…", "Uninstalling Dotsmith…")
                },
                tr!("Espera mientras se completa el proceso.", "Please wait while the process completes."),
            ))
            .child(div().flex().flex_col().gap(px(10.)).children(steps.iter().enumerate().map(|(i, (_, text))| {
                let (mark, color) = match i.cmp(&current) {
                    std::cmp::Ordering::Less => (icon("check").size(px(14.)).text_color(rgb(GREEN())).into_any_element(), TEXT()),
                    std::cmp::Ordering::Equal => (spinner(i).into_any_element(), TEXT()),
                    std::cmp::Ordering::Greater => (div().size(px(6.)).mx(px(4.)).rounded_full().bg(rgb(BORDER())).into_any_element(), MUTED_2()),
                };
                div()
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .child(div().w(px(16.)).flex().justify_center().child(mark))
                    .child(div().text_size(px(13.)).text_color(rgb(color)).child(text.to_string()))
            })))
            .when(show_bar, |d| {
                d.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(8.))
                        .child(
                            div().h(px(8.)).w_full().rounded_full().bg(rgb(HOVER())).child(div().h_full().w(relative(frac)).rounded_full().bg(rgb(ACCENT()))),
                        )
                        .child(
                            div()
                                .flex()
                                .gap(px(12.))
                                .text_size(px(11.5))
                                .text_color(rgb(MUTED_2()))
                                .child(div().flex_1().min_w_0().truncate().font_family(MONO).child(p.file.clone()))
                                .child(format!("{} / {}", bytes(p.done), bytes(p.total))),
                        ),
                )
            })
    }

    fn done(&self, cx: &mut Context<Self>) -> Div {
        let (title, text) = if self.installing() {
            (tr!("Instalación completada", "Installation complete"), tr!("Dotsmith se ha instalado correctamente.", "Dotsmith was installed successfully."))
        } else {
            (tr!("Desinstalación completada", "Uninstall complete"), tr!("Dotsmith se ha quitado de este equipo.", "Dotsmith was removed from this computer."))
        };
        div().flex().flex_col().gap(px(14.)).child(icon("circle-check-big").size(px(34.)).text_color(rgb(GREEN()))).child(heading(title, text)).when(
            self.installing(),
            |d| {
                d.child(toggle("launch", tr!("Ejecutar Dotsmith", "Run Dotsmith"), self.launch).on_click(cx.listener(|s, _, _, cx| {
                    s.launch = !s.launch;
                    cx.notify();
                })))
            },
        )
    }

    fn failed(&self, message: &str) -> Div {
        let title = if self.installing() {
            tr!("No se pudo instalar", "Dotsmith could not be installed")
        } else {
            tr!("No se pudo desinstalar", "Dotsmith could not be uninstalled")
        };
        div().flex().flex_col().gap(px(14.)).child(icon("triangle-alert").size(px(30.)).text_color(rgb(DANGER()))).child(heading(title, "")).child(
            div()
                .p(px(12.))
                .rounded(px(8.))
                .bg(rgb(INPUT_BG()))
                .border_1()
                .border_color(rgb(BORDER_SOFT()))
                .text_size(px(12.))
                .text_color(rgb(MUTED()))
                .child(message.to_string()),
        )
    }

    fn license(&self) -> Div {
        div().flex().flex_col().gap(px(12.)).h_full().child(heading(tr!("Licencia", "License"), "MIT")).child(
            div()
                .flex_1()
                .min_h_0()
                .p(px(12.))
                .rounded(px(8.))
                .bg(rgb(INPUT_BG()))
                .border_1()
                .border_color(rgb(BORDER_SOFT()))
                .text_size(px(11.5))
                .font_family(MONO)
                .text_color(rgb(MUTED()))
                .child(include_str!("../../LICENSE")),
        )
    }

    fn footer(&self, cx: &mut Context<Self>) -> Div {
        let row = div().flex().items_center().justify_end().gap(px(8.));
        match &self.screen {
            Screen::Choose if self.installing() => {
                let go = match self.action() {
                    Action::Install => tr!("Instalar", "Install"),
                    Action::Update => tr!("Actualizar", "Update"),
                    Action::Reinstall => tr!("Reinstalar", "Reinstall"),
                };
                row.child(button("cancel", tr!("Cancelar", "Cancel"), None).on_click(cx.listener(|s, _, w, cx| s.back(w, cx))))
                    .child(primary("go", go, false).on_click(cx.listener(|s, _, w, cx| s.primary(w, cx))))
            }
            Screen::Choose => row
                .child(button("cancel", tr!("Cancelar", "Cancel"), None).on_click(cx.listener(|s, _, w, cx| s.back(w, cx))))
                .child(primary("go", tr!("Desinstalar", "Uninstall"), true).on_click(cx.listener(|s, _, w, cx| s.primary(w, cx)))),
            Screen::Working => row,
            Screen::Done => {
                let text = if self.installing() { tr!("Terminar", "Finish") } else { tr!("Cerrar", "Close") };
                row.child(primary("finish", text, false).on_click(cx.listener(|s, _, w, _| s.finish(w))))
            }
            Screen::Failed(_) => {
                let retry = !matches!(self.mode, Mode::Install(None));
                row.child(button("close", tr!("Cerrar", "Close"), None).on_click(cx.listener(|_, _, w, _| w.remove_window())))
                    .when(retry, |d| d.child(primary("retry", tr!("Reintentar", "Try again"), false).on_click(cx.listener(|s, _, _, cx| s.start(cx)))))
            }
            Screen::License => row.child(primary("back", tr!("Volver", "Back"), false).on_click(cx.listener(|s, _, w, cx| s.primary(w, cx)))),
        }
    }

    fn titlebar(&self) -> Div {
        let busy = self.screen == Screen::Working;
        let caption = |id: &'static str, name: &'static str, hover: u32| {
            div()
                .id(id)
                .group(id)
                .w(px(46.))
                .h_full()
                .flex()
                .items_center()
                .justify_center()
                .hover(move |s| s.bg(rgb(hover)))
                .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                .child(icon(name).size(px(15.)).text_color(rgb(MUTED())).group_hover(id, |s| s.text_color(rgb(TEXT()))))
        };
        div()
            .h(px(TITLE_H))
            .flex_none()
            .flex()
            .items_center()
            .bg(rgb(BG()))
            .border_b_1()
            .border_color(rgb(BORDER_SOFT()))
            .child(
                div()
                    .id("drag")
                    .flex_1()
                    .h_full()
                    .flex()
                    .items_center()
                    .gap(px(9.))
                    .pl(px(14.))
                    .window_control_area(WindowControlArea::Drag)
                    .on_mouse_down(MouseButton::Left, |_: &MouseDownEvent, window, cx| {
                        begin_native_move(window);
                        cx.stop_propagation();
                    })
                    .child(img(self.logo.clone()).w(px(20.)).h(px(20.)))
                    .child(div().text_size(px(13.)).font_weight(FontWeight::BOLD).child("Dotsmith"))
                    .child(div().text_size(px(12.)).text_color(rgb(MUTED_2())).child(if self.installing() {
                        tr!("Instalador", "Setup")
                    } else {
                        tr!("Desinstalador", "Uninstall")
                    })),
            )
            .child(caption("caption-min", "minus", HOVER()).on_click(|_, window, _| window.minimize_window()))
            .when(!busy, |d| d.child(caption("caption-close", "x", DANGER()).on_click(|_, window, _| window.remove_window())))
    }
}

impl Render for Setup {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let body = match self.screen.clone() {
            Screen::Choose => self.choose(cx),
            Screen::Working => self.working(),
            Screen::Done => self.done(cx),
            Screen::Failed(m) => self.failed(&m),
            Screen::License => self.license(),
        };
        div()
            .track_focus(&self.focus)
            .on_action(cx.listener(|s, _: &Primary, w, cx| s.primary(w, cx)))
            .on_action(cx.listener(|s, _: &Back, w, cx| s.back(w, cx)))
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(BG()))
            .text_color(rgb(TEXT()))
            .font_family(FONT)
            .child(self.titlebar())
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .child(
                        div()
                            .w(px(HERO_W as f32))
                            .h_full()
                            .flex_none()
                            .relative()
                            .border_r_1()
                            .border_color(rgb(BORDER_SOFT()))
                            .child(img(self.hero.clone()).absolute().top_0().left_0().size_full())
                            .child(
                                div()
                                    .absolute()
                                    .left_0()
                                    .right_0()
                                    .bottom(px(26.))
                                    .flex()
                                    .flex_col()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(div().text_size(px(22.)).font_weight(FontWeight::BOLD).child("Dotsmith"))
                                    .child(
                                        div()
                                            .px(px(8.))
                                            .py(px(2.))
                                            .rounded_full()
                                            .bg(rgb(RAISED()))
                                            .border_1()
                                            .border_color(rgb(BORDER_SOFT()))
                                            .text_size(px(11.))
                                            .font_family(MONO)
                                            .text_color(rgb(MUTED()))
                                            .child(format!("v{VERSION}")),
                                    )
                                    .child(div().pt(px(6.)).text_size(px(11.)).text_color(rgb(MUTED_2())).child("Michael Araque")),
                            ),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .flex()
                            .flex_col()
                            .px(px(30.))
                            .pt(px(28.))
                            .pb(px(22.))
                            .child(div().flex_1().min_h_0().child(body))
                            .child(self.footer(cx)),
                    ),
            )
    }
}

// ---------------------------------------------------------------------- widgets

fn heading(title: impl Into<SharedString>, sub: impl Into<SharedString>) -> Div {
    let sub: SharedString = sub.into();
    div()
        .flex()
        .flex_col()
        .gap(px(4.))
        .child(div().text_size(px(19.)).font_weight(FontWeight::BOLD).child(title.into()))
        .when(!sub.is_empty(), |d| d.child(div().text_size(px(12.5)).text_color(rgb(MUTED())).child(sub)))
}

fn label(text: &str) -> Div {
    div().text_size(px(11.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(MUTED())).child(text.to_string())
}

fn note(text: impl Into<SharedString>) -> Div {
    div().p(px(12.)).rounded(px(8.)).bg(rgb(PANEL())).border_1().border_color(rgb(BORDER_SOFT())).text_size(px(12.5)).child(text.into())
}

fn toggle(id: &'static str, text: &str, on: bool) -> Stateful<Div> {
    div()
        .id(id)
        .flex()
        .items_center()
        .gap(px(10.))
        .py(px(6.))
        .cursor_pointer()
        .hover(|s| s.opacity(0.9))
        .child(
            div()
                .w(px(30.))
                .h(px(17.))
                .flex_none()
                .rounded_full()
                .p(px(2.))
                .flex()
                .when(on, |d| d.bg(rgb(ACCENT())).justify_end())
                .when(!on, |d| d.bg(rgb(HOVER())).justify_start())
                .child(div().w(px(13.)).h(px(13.)).rounded_full().bg(rgb(if on { ACCENT_INK() } else { MUTED() }))),
        )
        .child(div().text_size(px(12.5)).child(text.to_string()))
}

fn button(id: &'static str, text: &str, name: Option<&'static str>) -> Stateful<Div> {
    div()
        .id(id)
        .flex_none()
        .px(px(14.))
        .h(px(32.))
        .flex()
        .items_center()
        .justify_center()
        .gap(px(6.))
        .rounded(px(7.))
        .bg(rgb(RAISED()))
        .border_1()
        .border_color(rgb(BORDER()))
        .text_size(px(12.5))
        .cursor_pointer()
        .hover(|s| s.bg(rgb(HOVER())))
        .when_some(name, |d, n| d.child(icon(n).size(px(14.)).text_color(rgb(MUTED()))))
        .child(text.to_string())
}

fn primary(id: &'static str, text: &str, danger: bool) -> Stateful<Div> {
    let (bg, hover, ink) = if danger { (DANGER(), DANGER(), 0xffffff) } else { (ACCENT(), ACCENT_HOVER(), ACCENT_INK()) };
    div()
        .id(id)
        .flex_none()
        .px(px(20.))
        .h(px(32.))
        .flex()
        .items_center()
        .justify_center()
        .rounded(px(7.))
        .bg(rgb(bg))
        .text_color(rgb(ink))
        .text_size(px(12.5))
        .font_weight(FontWeight::SEMIBOLD)
        .cursor_pointer()
        .hover(move |s| s.bg(rgb(hover)).opacity(0.92))
        .child(text.to_string())
}

fn bytes(n: u64) -> String {
    if n >= 1 << 30 { format!("{:.1} GB", n as f64 / (1u64 << 30) as f64) } else { format!("{} MB", n >> 20) }
}

// ---------------------------------------------------------------------- art

fn logo() -> RgbaImage {
    image::load_from_memory(include_bytes!("../../gpui/assets/dotsmith-64.png")).expect("logo").to_rgba8()
}

/// RGBA > the BGRA frame gpui draws, scaled `k` times with hard pixel edges.
fn render(im: &RgbaImage, k: u32) -> Arc<RenderImage> {
    let mut im = if k > 1 { image::imageops::resize(im, im.width() * k, im.height() * k, image::imageops::FilterType::Nearest) } else { im.clone() };
    for p in im.as_chunks_mut::<4>().0 {
        p.swap(0, 2);
    }
    Arc::new(RenderImage::new(vec![image::Frame::new(im)]))
}

/// The left panel: the transparency checkerboard every pixel-art editor has, fading into the panel colour, with the
/// logo drawn three times its size.
fn hero() -> Arc<RenderImage> {
    let h = (H - TITLE_H) as u32;
    let rgb_of = |c: u32| [(c >> 16) as u8, (c >> 8) as u8, c as u8];
    let (a, b, panel) = (rgb_of(CHECK_A()), rgb_of(CHECK_B()), rgb_of(PANEL()));
    let mut im = RgbaImage::new(HERO_W, h);
    for (x, y, p) in im.enumerate_pixels_mut() {
        let c = if (x / 12 + y / 12) % 2 == 0 { a } else { b };
        let t = ((y as f32 / h as f32 - 0.45) / 0.3).clamp(0., 1.);
        let mix = |i: usize| (c[i] as f32 * (1. - t) + panel[i] as f32 * t) as u8;
        *p = image::Rgba([mix(0), mix(1), mix(2), 255]);
    }
    let logo = image::imageops::resize(&logo(), 64 * 3, 64 * 3, image::imageops::FilterType::Nearest);
    image::imageops::overlay(&mut im, &logo, ((HERO_W - logo.width()) / 2) as i64, 54);
    render(&im, 1)
}

// ---------------------------------------------------------------------- icons

macro_rules! icons {
    ($($name:literal),* $(,)?) => {
        const ICONS: &[(&str, &[u8])] = &[$(($name, include_bytes!(concat!("../../gpui/assets/icons/", $name, ".svg")))),*];
    };
}

icons!("check", "circle-check-big", "folder-open", "loader-circle", "minus", "triangle-alert", "x");

struct Icons;

impl AssetSource for Icons {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        let name = path.strip_prefix("icons/").and_then(|p| p.strip_suffix(".svg"));
        Ok(name.and_then(|n| ICONS.iter().find(|i| i.0 == n)).map(|i| Cow::Borrowed(i.1)))
    }

    fn list(&self, path: &str) -> gpui::Result<Vec<SharedString>> {
        Ok(ICONS.iter().map(|i| format!("icons/{}.svg", i.0)).filter(|p| p.starts_with(path)).map(SharedString::from).collect())
    }
}

fn icon(name: &'static str) -> Svg {
    svg().path(SharedString::from(format!("icons/{name}.svg"))).flex_none().size(px(16.))
}

fn spinner(i: usize) -> impl IntoElement {
    use gpui::{Animation, AnimationExt, Transformation, percentage};
    icon("loader-circle")
        .size(px(14.))
        .text_color(rgb(ACCENT_DIM()))
        .with_animation(("spin", i), Animation::new(Duration::from_secs(1)).repeat(), |s, d| s.with_transformation(Transformation::rotate(percentage(d))))
}

/// Hand a title-bar drag to Windows (gpui's own caption hit test misses on a transparent title bar).
fn begin_native_move(window: &Window) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    unsafe extern "system" {
        fn ReleaseCapture() -> i32;
        fn PostMessageW(hwnd: isize, msg: u32, wparam: usize, lparam: isize) -> i32;
    }
    if let Ok(h) = HasWindowHandle::window_handle(window)
        && let RawWindowHandle::Win32(w) = h.as_raw()
    {
        unsafe {
            ReleaseCapture();
            PostMessageW(w.hwnd.get(), 0x00A1 /* WM_NCLBUTTONDOWN */, 2 /* HTCAPTION */, 0);
        }
    }
}

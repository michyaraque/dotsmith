//! The model picker: a centred modal to choose base model (family + base/turbo) and LoRA checkpoint.
//! Everything it lists comes from the backend's "state" event (engine registry + catalog); nothing is hardcoded here
//! except how a family looks (its provider logo). Choices are staged and only sent when applied.
use gpui::{AnyElement, Context, FontWeight, ObjectFit, RenderImage, SharedString, div, img, prelude::*, px, rgb, rgba};
use serde_json::{Value, json};
use std::sync::Arc;

use super::{Keyboard, Studio, button, button_i, label};
use crate::icons::icon;
use crate::theme::*;
use crate::tr;

/// One model family as the backend describes it.
#[derive(Clone, Default)]
pub struct FamilyInfo {
    pub id: String,
    pub label: String,
    /// Hugging Face repo of the model card (author, avatar, description, stats).
    pub repo: String,
    pub scale: u32,
    pub alpha: bool,
    pub capabilities: Vec<String>,
    /// square sprite sizes it draws (empty = any)
    pub sizes: Vec<u32>,
    /// studio modes it supports (empty = all)
    pub modes: Vec<String>,
    /// reads the description as written (no camera or layout words)
    pub plain: bool,
    /// can paint an opaque background on request
    pub opaque: bool,
    /// oldest GPU it runs on (compute capability); None = any card this build supports
    pub min_compute: Option<(i32, i32)>,
    /// default steps of the base and the Turbo variant
    pub steps: [u32; 2],
    /// (base-model label sent back to the backend, short name shown, is turbo)
    pub variants: Vec<(String, String, bool)>,
}

impl FamilyInfo {
    pub fn parse(v: &Value) -> Vec<FamilyInfo> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .map(|f| FamilyInfo {
                        id: f["id"].as_str().unwrap_or("").into(),
                        label: f["label"].as_str().unwrap_or("").into(),
                        repo: f["repo"].as_str().unwrap_or("").into(),
                        scale: f["scale"].as_u64().unwrap_or(16) as u32,
                        alpha: f["alpha"].as_bool().unwrap_or(true),
                        capabilities: f["capabilities"].as_array().map(|t| t.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default(),
                        sizes: f["sizes"].as_array().map(|t| t.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect()).unwrap_or_default(),
                        modes: f["modes"].as_array().map(|t| t.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default(),
                        plain: f["plain"].as_bool().unwrap_or(false),
                        opaque: f["opaque"].as_bool().unwrap_or(true),
                        min_compute: f["min_compute"].as_array().and_then(|c| Some((c.first()?.as_i64()? as i32, c.get(1)?.as_i64()? as i32))),
                        steps: [false, true].map(|t| {
                            f["variants"]
                                .as_array()
                                .and_then(|vs| vs.iter().find(|x| x["turbo"].as_bool() == Some(t)))
                                .and_then(|x| x["steps"].as_u64())
                                .unwrap_or(if t { 8 } else { 25 }) as u32
                        }),
                        variants: f["variants"]
                            .as_array()
                            .map(|vs| {
                                vs.iter()
                                    .map(|x| {
                                        (
                                            x["label"].as_str().unwrap_or("").to_string(),
                                            x["name"].as_str().unwrap_or("").to_string(),
                                            x["turbo"].as_bool().unwrap_or(false),
                                        )
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// What the user has selected in the open picker, before applying.
#[derive(Clone, Default)]
pub struct PickerState {
    pub family: String,
    pub base: String,         // base-model label (family + variant)
    pub run: Option<String>,  // run folder
    pub ckpt: Option<String>, // checkpoint path (None = no LoRA)
}

/// What the catalog says about a family's output, as short facts: transparency and pixel scale.
pub fn traits(f: &FamilyInfo) -> Vec<String> {
    vec![if f.alpha { tr!("Alfa nativo", "Native alpha").to_string() } else { tr!("Fondo con clave", "Keyed background").to_string() }, format!("x{}", f.scale)]
}

/// A checkpoint label from the catalog: a bare step number reads as "step N".
fn ckpt_label(l: &str) -> String {
    if !l.is_empty() && l.chars().all(|c| c.is_ascii_digit()) { format!("{} {l}", tr!("paso", "step")) } else { l.to_string() }
}

fn pill(text: impl Into<SharedString>) -> gpui::Div {
    div()
        .px(px(7.))
        .h(px(20.))
        .flex()
        .items_center()
        .rounded(px(5.))
        .bg(rgb(BG()))
        .border_1()
        .border_color(rgb(BORDER_SOFT()))
        .text_size(px(10.5))
        .text_color(rgb(MUTED()))
        .child(text.into())
}

fn primary(id: &'static str, text: &str, enabled: bool) -> gpui::Stateful<gpui::Div> {
    div()
        .kb(id)
        .px(px(16.))
        .h(px(32.))
        .flex()
        .items_center()
        .justify_center()
        .rounded(px(7.))
        .text_size(px(12.5))
        .font_weight(FontWeight::SEMIBOLD)
        .when(enabled, |d| d.bg(rgb(ACCENT())).text_color(rgb(ACCENT_INK())).cursor_pointer().hover(|s| s.bg(rgb(ACCENT_HOVER()))))
        .when(!enabled, |d| d.bg(rgb(RAISED())).text_color(rgb(MUTED_2())))
        .child(text.to_string())
}

/// A thin bar: `Some(f)` fills to f; `None` (no previous load to compare with) slides a short segment back and forth.
pub fn progress_bar(frac: Option<f32>, width: f32) -> AnyElement {
    let track = div().w(px(width)).h(px(4.)).flex_none().rounded_full().bg(rgb(RAISED())).overflow_hidden().relative();
    match frac {
        Some(f) => track.child(div().h_full().w(px(width * f.clamp(0.02, 1.))).rounded_full().bg(rgb(WARN()))).into_any_element(),
        None => {
            use gpui::{Animation, AnimationExt};
            let seg = width * 0.3;
            track
                .child(div().absolute().top_0().h_full().w(px(seg)).rounded_full().bg(rgb(WARN())).with_animation(
                    "bar-indet",
                    Animation::new(std::time::Duration::from_millis(1200)).repeat(),
                    move |d, t| {
                        let x = if t < 0.5 { t * 2. } else { 2. - t * 2. };
                        d.left(px((width - seg) * x))
                    },
                ))
                .into_any_element()
        }
    }
}

impl Studio {
    /// The family's publisher avatar, from Hugging Face.
    pub(super) fn logo(&self, family: &str) -> Option<Arc<RenderImage>> {
        self.model.families.iter().find(|f| f.id == family).and_then(|f| self.hf_info(&f.repo)).and_then(|(_, a)| a)
    }

    /// Publisher and stats line of a family, from Hugging Face (the repo owner until it loads).
    pub(super) fn publisher(&self, f: &FamilyInfo) -> String {
        match self.hf_info(&f.repo) {
            Some((i, _)) => i.author.clone(),
            None => f.repo.split('/').next().unwrap_or("").to_string(),
        }
    }

    /// One click on a recent model: switch to it and put it on the GPU.
    fn switch_to(&mut self, base: &str, lora: &str) {
        let lora = if lora.is_empty() { "-".to_string() } else { lora.to_string() };
        self.send(json!({"cmd": "select", "base": base, "lora": lora, "load": true}));
        self.model_open = false;
    }

    /// The model selector in the middle of the title bar: a fixed-width bar with the base model and LoRA in use. While
    /// a model loads it shows the progress (what, bar, time), and a switch waiting behind a job says so.
    pub(super) fn model_chip(&self, cx: &mut Context<Self>) -> AnyElement {
        let shell = div()
            .kb("model-chip")
            .w(px(440.))
            .h(px(32.))
            .flex()
            .items_center()
            .gap(px(10.))
            .pl(px(8.))
            .pr(px(10.))
            .rounded(px(9.))
            .cursor_pointer()
            .bg(rgb(RAISED()))
            .border_1()
            .border_color(rgb(if self.model_open { ACCENT_DIM() } else { BORDER() }))
            .hover(|s| s.bg(rgb(HOVER())).border_color(rgb(ACCENT_DIM())))
            .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .on_click(cx.listener(|this, _, _, cx| {
                if this.model_open {
                    this.model_open = false
                } else {
                    this.open_model_picker()
                }
                cx.notify();
            }));
        let tail = || {
            div().flex_none().flex().items_center().gap(px(8.)).child(super::kbd("Ctrl L")).child(icon("chevron-down").size(px(13.)).text_color(rgb(MUTED())))
        };
        if let Some((what, frac, txt)) = self.load_progress() {
            return shell
                .child(div().w(px(20.)).flex().justify_center().child(crate::icons::spinner("chip-spin", 13., WARN())))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .overflow_hidden()
                        .text_ellipsis()
                        .whitespace_nowrap()
                        .text_size(px(12.))
                        .text_color(rgb(TEXT()))
                        .child(tr!(format!("Cargando {what}"), format!("Loading {what}"))),
                )
                .child(progress_bar(frac, 64.))
                .child(div().flex_none().text_size(px(11.)).text_color(rgb(MUTED_2())).child(txt))
                .into_any_element();
        }
        if let Some((_, label)) = &self.activity.queued {
            let after = if self.running { tr!("tras la generación", "after the generation") } else { tr!("en cola", "queued") };
            return shell
                .child(div().w(px(20.)).flex().justify_center().child(crate::icons::spinner("chip-spin", 13., MUTED_2())))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .overflow_hidden()
                        .text_ellipsis()
                        .whitespace_nowrap()
                        .text_size(px(12.))
                        .text_color(rgb(TEXT()))
                        .child(format!("{} {label}", tr!("Luego:", "Next:"))),
                )
                .child(div().flex_none().text_size(px(11.)).text_color(rgb(MUTED_2())).child(after))
                .into_any_element();
        }
        let m = &self.model;
        let step = crate::backend::checkpoint_name(std::path::Path::new(&m.lora));
        let has_loras = m.runs.iter().any(|r| r.family == m.family);
        let lora = if !m.lora.is_empty() {
            format!("{} · {}", m.version, step)
        } else if has_loras {
            tr!("sin LoRA", "no LoRA").to_string()
        } else {
            String::new()
        };
        let logo = match self.logo(&m.family) {
            Some(l) => div().w(px(20.)).h(px(20.)).flex_none().rounded(px(5.)).overflow_hidden().child(img(l).size_full().object_fit(ObjectFit::Cover)),
            None => div()
                .w(px(20.))
                .h(px(20.))
                .flex_none()
                .rounded(px(5.))
                .bg(rgb(HOVER()))
                .flex()
                .items_center()
                .justify_center()
                .child(icon("box").size(px(12.)).text_color(rgb(MUTED()))),
        };
        let title = if m.base.is_empty() { tr!("Elegir modelo", "Choose a model").to_string() } else { m.base.clone() };
        shell
            .child(logo)
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .overflow_hidden()
                    .child(div().flex_none().text_size(px(13.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(title))
                    .when(!m.base.is_empty(), |d| {
                        d.child(div().min_w_0().overflow_hidden().text_ellipsis().whitespace_nowrap().text_size(px(12.)).text_color(rgb(MUTED())).child(lora))
                    }),
            )
            .child(tail())
            .into_any_element()
    }

    /// Open the picker on the current selection.
    pub(super) fn open_model_picker(&mut self) {
        let m = &self.model;
        self.picker = PickerState {
            family: m.family.clone(),
            base: m.base.clone(),
            run: (!m.run.is_empty()).then(|| m.run.clone()),
            ckpt: (!m.lora.is_empty()).then(|| m.lora.clone()),
        };
        self.model_open = true;
    }

    fn pick_family(&mut self, f: &FamilyInfo) {
        if self.picker.family == f.id {
            return;
        }
        self.picker.family = f.id.clone();
        self.picker.base = f.variants.first().map(|v| v.0.clone()).unwrap_or_default();
        // the newest run of that family and its newest checkpoint, like the backend would pick
        let run = self.model.runs.iter().find(|r| r.family == f.id);
        self.picker.run = run.map(|r| r.run.clone());
        self.picker.ckpt = run.and_then(|r| r.ckpts.first().map(|c| c.1.clone()));
    }

    pub(super) fn model_picker(&mut self, cx: &mut Context<Self>) -> AnyElement {
        if self.picker.family.is_empty() && !self.model.family.is_empty() {
            self.open_model_picker(); // opened before the first state arrived (--open-model)
        }
        let families = self.model.families.clone();
        let pk = self.picker.clone();
        let runs: Vec<_> = self.model.runs.iter().filter(|r| r.family == pk.family).cloned().collect();
        let fam_label = families.iter().find(|f| f.id == pk.family).map(|f| f.label.clone()).unwrap_or_default();
        let loaded = self.model.loaded.clone();
        let changed = pk.base != self.model.base || pk.ckpt.clone().unwrap_or_default() != self.model.lora;
        let family_change = pk.family != self.model.family;

        // ---------------------------------------------------------------- left: base models
        let cards = families.into_iter().enumerate().map(|(i, f)| {
            let sel = f.id == pk.family;
            let fits = self.model.runs_here(&f);
            let is_loaded = loaded.as_ref().is_some_and(|l| f.variants.iter().any(|v| &v.0 == l));
            let f2 = f.clone();
            let variants = f.variants.clone();
            div()
                .kb(("fam", i))
                .p(px(12.))
                .rounded(px(10.))
                .border_1()
                .cursor_pointer()
                .flex()
                .flex_col()
                .gap(px(10.))
                .when(sel, |d| d.bg(rgb(RAISED())).border_color(rgb(ACCENT_DIM())))
                .when(!sel, |d| d.bg(rgb(PANEL())).border_color(rgb(BORDER_SOFT())).hover(|s| s.bg(rgb(RAISED())).border_color(rgb(BORDER()))))
                .when(!fits, |d| d.opacity(0.45).cursor_default())
                .when(fits, |d| {
                    d.on_click(cx.listener(move |this, _, _, cx| {
                        this.pick_family(&f2);
                        cx.notify();
                    }))
                })
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(12.))
                        .child(match self.logo(&f.id) {
                            Some(l) => {
                                div().w(px(40.)).h(px(40.)).flex_none().rounded(px(9.)).overflow_hidden().child(img(l).size_full().object_fit(ObjectFit::Cover))
                            }
                            None => div().w(px(40.)).h(px(40.)).flex_none().rounded(px(9.)).bg(rgb(HOVER())),
                        })
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .flex()
                                .flex_col()
                                .gap(px(2.))
                                .child(div().text_size(px(14.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(f.label.clone()))
                                .child(div().text_size(px(11.5)).text_color(rgb(MUTED_2())).child(if fits {
                                    self.publisher(&f)
                                } else {
                                    super::ModelState::needs_newer(&f)
                                })),
                        )
                        .when(is_loaded, |d| {
                            d.child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(5.))
                                    .px(px(7.))
                                    .h(px(20.))
                                    .rounded_full()
                                    .bg(rgba(0xe5a95f22))
                                    .text_size(px(10.5))
                                    .text_color(rgb(BLUE()))
                                    .child(div().w(px(6.)).h(px(6.)).rounded_full().bg(rgb(BLUE())))
                                    .child(tr!("En GPU", "On GPU")),
                            )
                        }),
                )
                .when(sel && variants.len() > 1, |d| {
                    d.child(div().flex().p(px(3.)).gap(px(3.)).rounded(px(8.)).bg(rgb(BG())).children(variants.into_iter().enumerate().map(
                        |(j, (lab, name, turbo))| {
                            let on = lab == pk.base;
                            let lab2 = lab.clone();
                            div()
                                .kb(("var", i * 10 + j))
                                .flex_1()
                                .h(px(30.))
                                .flex()
                                .items_center()
                                .justify_center()
                                .gap(px(6.))
                                .rounded(px(6.))
                                .text_size(px(12.))
                                .cursor_pointer()
                                .when(on, |d| d.bg(rgb(ACCENT())).text_color(rgb(ACCENT_INK())).font_weight(FontWeight::SEMIBOLD))
                                .when(!on, |d| d.text_color(rgb(MUTED())).hover(|s| s.bg(rgb(HOVER())).text_color(rgb(TEXT()))))
                                .when(turbo, |d| d.child(icon("zap").size(px(13.)).text_color(rgb(if on { ACCENT_INK() } else { MUTED_2() }))))
                                .child(if turbo { format!("{name} · {}", tr!("8 pasos", "8 steps")) } else { name })
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.picker.base = lab2.clone();
                                    cx.notify();
                                }))
                        },
                    )))
                })
        });

        // ---------------------------------------------------------------- right: LoRAs of that family
        let no_lora = pk.ckpt.is_none();
        let none_card = div()
            .kb("nolora")
            .px(px(12.))
            .py(px(10.))
            .flex()
            .items_center()
            .gap(px(10.))
            .rounded(px(9.))
            .border_1()
            .cursor_pointer()
            .when(no_lora, |d| d.bg(rgb(RAISED())).border_color(rgb(ACCENT_DIM())))
            .when(!no_lora, |d| d.bg(rgb(PANEL())).border_color(rgb(BORDER_SOFT())).hover(|s| s.border_color(rgb(BORDER()))))
            .child(
                div()
                    .w(px(30.))
                    .h(px(30.))
                    .flex_none()
                    .rounded(px(7.))
                    .bg(rgb(BG()))
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(icon("box").size(px(15.)).text_color(rgb(if no_lora { ACCENT() } else { MUTED_2() }))),
            )
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap(px(1.))
                    .child(div().text_size(px(13.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(tr!("Sin LoRA", "No LoRA")))
                    .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(tr!("Solo el modelo base", "The base model on its own"))),
            )
            .on_click(cx.listener(|this, _, _, cx| {
                this.picker.ckpt = None;
                this.picker.run = None;
                cx.notify();
            }));
        let run_list = if runs.is_empty() {
            div()
                .flex_1()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(none_card)
                .child(
                    div()
                        .flex_1()
                        .flex()
                        .flex_col()
                        .items_center()
                        .justify_center()
                        .gap(px(8.))
                        .child(icon("layers").size(px(26.)).text_color(rgb(MUTED_2())))
                        .child(
                            div()
                                .text_size(px(13.))
                                .text_color(rgb(MUTED()))
                                .child(tr!(format!("Aún no hay LoRAs de {fam_label}"), format!("No {fam_label} LoRAs yet"))),
                        )
                        .child(
                            div()
                                .text_size(px(11.5))
                                .text_color(rgb(MUTED_2()))
                                .child(tr!("Copia archivos .safetensors en tu carpeta de LoRAs.", "Copy .safetensors files into your LoRA folder.")),
                        )
                        .when_some(self.settings.lora_library(), |d, dir| {
                            d.child(div().pt(px(6.)).child(
                                button_i("open-loras", "folder-open", tr!("Abrir la carpeta de LoRAs", "Open the LoRA folder")).on_click(cx.listener(
                                    move |_, _, _, cx| {
                                        let _ = std::fs::create_dir_all(&dir);
                                        cx.open_with_system(&dir);
                                    },
                                )),
                            ))
                        }),
                )
                .into_any_element()
        } else {
            div()
                .id("runs")
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .flex()
                .flex_col()
                .gap(px(6.))
                .pr(px(4.))
                .child(none_card)
                .children(runs.into_iter().enumerate().map(|(i, r)| {
                    let sel = pk.run.as_deref() == Some(r.run.as_str());
                    let in_use = r.run == self.model.run;
                    let run2 = r.run.clone();
                    let first = r.ckpts.first().map(|c| c.1.clone());
                    let ckpts = r.ckpts.clone();
                    div()
                        .id(("run", i))
                        .rounded(px(9.))
                        .border_1()
                        .flex()
                        .flex_col()
                        .when(sel, |d| d.bg(rgb(RAISED())).border_color(rgb(ACCENT_DIM())))
                        .when(!sel, |d| d.bg(rgb(PANEL())).border_color(rgb(BORDER_SOFT())).hover(|s| s.border_color(rgb(BORDER()))))
                        .child(
                            div()
                                .kb(("runhead", i))
                                .px(px(12.))
                                .py(px(10.))
                                .flex()
                                .items_center()
                                .gap(px(10.))
                                .cursor_pointer()
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    if this.picker.run.as_deref() != Some(run2.as_str()) {
                                        this.picker.run = Some(run2.clone());
                                        this.picker.ckpt = first.clone();
                                    }
                                    cx.notify();
                                }))
                                .child(
                                    div()
                                        .w(px(30.))
                                        .h(px(30.))
                                        .flex_none()
                                        .rounded(px(7.))
                                        .bg(rgb(BG()))
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .child(icon("layers").size(px(15.)).text_color(rgb(if sel { ACCENT() } else { MUTED_2() }))),
                                )
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w_0()
                                        .flex()
                                        .flex_col()
                                        .gap(px(1.))
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap(px(6.))
                                                .child(
                                                    div().text_size(px(13.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(r.version.clone()),
                                                )
                                                .when(in_use, |d| d.child(div().text_size(px(10.)).text_color(rgb(BLUE())).child(tr!("· en uso", "· in use")))),
                                        )
                                        .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(if r.name.is_empty() {
                                            r.age.clone()
                                        } else {
                                            format!("{} · {}", r.name, r.age)
                                        })),
                                )
                                .child(
                                    div()
                                        .id(("scale", i))
                                        .child(pill(format!("x{}", r.scale)))
                                        .tooltip(super::tooltip(tr!("Píxeles de imagen por píxel de sprite", "Image pixels per sprite pixel"))),
                                )
                                .when(r.magenta, |d| {
                                    d.child(div().id(("key", i)).w(px(10.)).h(px(10.)).rounded(px(3.)).bg(rgb(0xff00ff)).tooltip(super::tooltip(tr!(
                                        "Dibuja sobre magenta; el fondo se quita solo",
                                        "Draws on magenta; the background is removed for you"
                                    ))))
                                }),
                        )
                        .when(sel, |d| {
                            d.child(div().px(px(12.)).pb(px(12.)).flex().flex_wrap().gap(px(5.)).children(ckpts.into_iter().enumerate().map(
                                |(j, (name, path))| {
                                    let on = pk.ckpt.as_deref() == Some(path.as_str());
                                    div()
                                        .kb(("ck", i * 100 + j))
                                        .px(px(10.))
                                        .h(px(26.))
                                        .flex()
                                        .items_center()
                                        .rounded(px(6.))
                                        .text_size(px(11.5))
                                        .cursor_pointer()
                                        .when(on, |d| d.bg(rgb(ACCENT())).text_color(rgb(ACCENT_INK())).font_weight(FontWeight::SEMIBOLD))
                                        .when(!on, |d| {
                                            d.bg(rgb(BG()))
                                                .text_color(rgb(MUTED()))
                                                .border_1()
                                                .border_color(rgb(BORDER_SOFT()))
                                                .hover(|s| s.text_color(rgb(TEXT())).border_color(rgb(BORDER())))
                                        })
                                        .child(ckpt_label(&name))
                                        .on_click(cx.listener(move |this, ev: &gpui::ClickEvent, _, cx| {
                                            this.picker.ckpt = Some(path.clone());
                                            if ev.click_count() >= 2 {
                                                let base = this.picker.base.clone();
                                                this.switch_to(&base, &path); // double click = use it now
                                            }
                                            cx.notify();
                                        }))
                                },
                            )))
                        })
                }))
                .into_any_element()
        };

        let hint = if family_change && loaded.is_some() {
            tr!("Otro modelo: se recarga entero.", "Another model: it reloads in full.")
        } else if pk.base != self.model.base && loaded.is_some() {
            tr!("Cambiar de variante tarda unos segundos.", "Switching variant takes a few seconds.")
        } else {
            ""
        };
        let progress = self.load_progress();
        let queued = self.activity.queued.clone();
        let busy = progress.is_some() || queued.is_some();

        // ---------------------------------------------------------------- recent: one click switches
        let cur_lora = self.model.lora.clone();
        let cur_base = self.model.base.clone();
        let recent: Vec<_> = self.model.recent.iter().filter(|r| !(r.0 == cur_base && r.1 == cur_lora)).take(4).cloned().collect();
        let recent_row = (!recent.is_empty()).then(|| {
            div()
                .px(px(22.))
                .py(px(10.))
                .flex()
                .items_center()
                .gap(px(8.))
                .overflow_hidden()
                .border_b_1()
                .border_color(rgb(BORDER_SOFT()))
                .bg(rgb(BG()))
                .child(icon("history").size(px(13.)).text_color(rgb(MUTED_2())))
                .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(tr!("Recientes", "Recent")))
                .children(recent.into_iter().enumerate().map(|(i, (base, lora, label, family))| {
                    let path = std::path::Path::new(&lora);
                    let ckpt = crate::backend::checkpoint_name(path);
                    let short = if lora.is_empty() {
                        tr!("sin LoRA", "no LoRA").to_string()
                    } else if path.parent().and_then(|p| p.file_name()).is_some_and(|run| path.file_stem() == Some(run))
                        || ckpt.chars().last().is_some_and(|c| c.is_ascii_digit())
                    {
                        let version = crate::jobs::version_of(path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                        format!("{version} · {ckpt}")
                    } else {
                        ckpt
                    };
                    let turbo = base.contains("Turbo");
                    div()
                        .kb(("recent", i))
                        .h(px(26.))
                        .max_w(px(240.))
                        .min_w_0()
                        .px(px(9.))
                        .flex()
                        .items_center()
                        .gap(px(6.))
                        .rounded(px(6.))
                        .cursor_pointer()
                        .tooltip(super::tooltip(&label))
                        .bg(rgb(PANEL()))
                        .border_1()
                        .border_color(rgb(BORDER_SOFT()))
                        .text_size(px(11.5))
                        .text_color(rgb(MUTED()))
                        .hover(|s| s.border_color(rgb(ACCENT_DIM())).text_color(rgb(TEXT())))
                        .child(match self.logo(&family) {
                            Some(l) => {
                                div().w(px(13.)).h(px(13.)).flex_none().rounded(px(3.)).overflow_hidden().child(img(l).size_full().object_fit(ObjectFit::Cover))
                            }
                            None => div().w(px(13.)).h(px(13.)).flex_none().rounded(px(3.)).bg(rgb(HOVER())),
                        })
                        .child(div().min_w_0().overflow_hidden().text_ellipsis().whitespace_nowrap().child(short))
                        .when(turbo, |d| d.child(icon("zap").size(px(11.)).text_color(rgb(ACCENT()))))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.switch_to(&base, &lora);
                            cx.notify();
                        }))
                }))
        });

        // ---------------------------------------------------------------- footer status: what the GPU is doing
        let footer_status: AnyElement = match (&progress, &queued) {
            (Some((what, frac, txt)), _) => div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap(px(10.))
                .child(crate::icons::spinner("foot-spin", 13., WARN()))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(4.))
                        .child(div().text_size(px(11.5)).text_color(rgb(TEXT())).child(tr!(format!("Cargando {what}"), format!("Loading {what}"))))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(8.))
                                .child(progress_bar(*frac, 180.))
                                .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(txt.clone())),
                        ),
                )
                .child(button("picker-cancel-load", tr!("Cancelar", "Cancel")).on_click(cx.listener(|this, _, _, cx| this.cancel_load(cx))))
                .into_any_element(),
            (None, Some((_, label))) => div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap(px(10.))
                .child(crate::icons::spinner("foot-spin", 13., MUTED_2()))
                .child(div().text_size(px(11.5)).text_color(rgb(MUTED())).child(tr!(
                    format!("En cola: {label} · {}", if self.running { "se cargará al terminar la generación" } else { "empieza enseguida" }),
                    format!("Queued: {label} · {}", if self.running { "loads when the generation ends" } else { "starts right away" })
                )))
                .into_any_element(),
            _ => div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap(px(10.))
                .when(!hint.is_empty(), |d| {
                    d.child(icon("info").size(px(13.)).text_color(rgb(MUTED_2())))
                        .child(div().flex_1().text_size(px(11.5)).text_color(rgb(MUTED_2())).child(hint))
                })
                .into_any_element(),
        };
        // primary: with a model on the GPU it switches; with none it applies and loads in one go
        let primary_label = if loaded.is_some() || busy { tr!("Cambiar modelo", "Switch model") } else { tr!("Cargar en GPU", "Load on GPU") };
        let primary_on = changed || (loaded.is_none() && !busy);

        let card = div()
            .id("picker")
            .occlude()
            .w(px(920.))
            .h(px(600.))
            .max_w_full()
            .max_h_full()
            .rounded(px(14.))
            .bg(rgb(PANEL()))
            .border_1()
            .border_color(rgb(BORDER()))
            .shadow_lg()
            .flex()
            .flex_col()
            .overflow_hidden()
            // header
            .child(
                div()
                    .px(px(22.))
                    .py(px(16.))
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .border_b_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .child(
                        div()
                            .w(px(34.))
                            .h(px(34.))
                            .rounded(px(9.))
                            .bg(rgb(RAISED()))
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(icon("box").size(px(17.)).text_color(rgb(ACCENT()))),
                    )
                    .child(div().flex_1().text_size(px(16.)).font_weight(FontWeight::BOLD).text_color(rgb(TEXT())).child(tr!("Modelo", "Model")))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(7.))
                            .px(px(10.))
                            .h(px(26.))
                            .rounded_full()
                            .bg(rgb(BG()))
                            .border_1()
                            .border_color(rgb(BORDER_SOFT()))
                            .child(div().w(px(7.)).h(px(7.)).rounded_full().bg(rgb(if loaded.is_some() { BLUE() } else { MUTED_2() })))
                            .child(div().text_size(px(11.5)).text_color(rgb(MUTED())).child(match &loaded {
                                Some(l) => format!("{}: {l}", tr!("En GPU", "On GPU")),
                                None => tr!("GPU libre", "GPU idle").into(),
                            })),
                    )
                    .child(button_i("picker-manage", "download", tr!("Descargar modelos", "Get models")).on_click(cx.listener(|this, _, _, cx| {
                        this.model_open = false;
                        this.open_hub(None, cx);
                    })))
                    .child(
                        div()
                            .kb("close")
                            .w(px(28.))
                            .h(px(28.))
                            .rounded(px(7.))
                            .flex()
                            .items_center()
                            .justify_center()
                            .cursor_pointer()
                            .hover(|s| s.bg(rgb(HOVER())))
                            .child(icon("x").size(px(15.)).text_color(rgb(MUTED())))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.model_open = false;
                                cx.notify();
                            })),
                    ),
            )
            .children(recent_row)
            // body
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .child(
                        div()
                            .w(px(350.))
                            .flex_none()
                            .p(px(16.))
                            .flex()
                            .flex_col()
                            .gap(px(10.))
                            .border_r_1()
                            .border_color(rgb(BORDER_SOFT()))
                            .child(label(tr!("Modelo base", "Base model")))
                            .children(cards),
                    )
                    .child(div().flex_1().min_w_0().p(px(16.)).flex().flex_col().gap(px(10.)).child(label(format!("LoRA · {fam_label}"))).child(run_list)),
            )
            // footer
            .child(
                div()
                    .px(px(22.))
                    .py(px(14.))
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .border_t_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .bg(rgb(BG()))
                    .child(footer_status)
                    .when(loaded.is_some() && !busy, |d| {
                        d.child(button("gpu", tr!("Liberar GPU", "Free GPU")).on_click(cx.listener(|this, _, _, cx| {
                            this.send(json!({"cmd": "unload"}));
                            cx.notify();
                        })))
                    })
                    .when(changed && loaded.is_none() && !busy, |d| {
                        d.child(button("apply-only", tr!("Elegir sin cargar", "Select without loading")).on_click(cx.listener(|this, _, _, cx| {
                            let p = this.picker.clone();
                            this.send(json!({"cmd": "select", "base": p.base, "lora": p.ckpt.unwrap_or_else(|| "-".into())}));
                            this.model_open = false;
                            cx.notify();
                        })))
                    })
                    .child(button("cancel", tr!("Cancelar", "Cancel")).on_click(cx.listener(|this, _, _, cx| {
                        this.model_open = false;
                        cx.notify();
                    })))
                    .child(primary("apply", primary_label, primary_on).when(primary_on, |d| {
                        d.on_click(cx.listener(|this, _, _, cx| {
                            let p = this.picker.clone();
                            this.switch_to(&p.base, &p.ckpt.unwrap_or_default()); // empty = no LoRA
                            cx.notify();
                        }))
                    })),
            );
        self.modal_layer("veil", card, |s, _, _| s.model_open = false, cx)
    }
}

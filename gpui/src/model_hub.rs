//! The model manager: every family in the catalog with its Hugging Face card (publisher, stats, description, README),
//! what is installed, and download options per variant. Plus the downloads panel under the title bar.
use gpui::{AnyElement, ClipboardItem, Context, FontWeight, ObjectFit, SharedString, div, img, prelude::*, px, rgb, rgba};
use serde_json::json;

use super::model_picker::{FamilyInfo, traits};
use super::{FileRow, Keyboard, Studio, button, button_i, icon_btn, label, tooltip};
use crate::download::gb;
use crate::hf::{self, Block, State};
use crate::icons::icon;
use crate::theme::*;
use crate::tr;

/// Bytes still to download for `rows` (unknown sizes count as zero), and whether every one has a source.
fn missing_bytes(s: &Studio, rows: &[FileRow]) -> (u64, bool) {
    let missing: Vec<&FileRow> = rows.iter().filter(|r| !r.found).collect();
    let bytes = missing.iter().filter_map(|r| r.remote.as_ref().and_then(|x| s.remote_size(x))).sum();
    (bytes, missing.iter().all(|r| r.remote.is_some()))
}

/// The files `variant` needs: the shared ones plus its own diffusion model.
fn variant_rows(rows: &[FileRow], turbo: bool) -> Vec<FileRow> {
    // precision alternatives (bf16 ...) are optional: downloaded one by one, never with the variant
    rows.iter().filter(|r| r.precision.is_none() && (r.turbo_only == turbo || !(r.role == "diffusion model"))).cloned().collect()
}

pub(super) fn avatar(image: Option<std::sync::Arc<gpui::RenderImage>>, text: &str, size: f32) -> gpui::Div {
    let tile = div().w(px(size)).h(px(size)).flex_none().rounded(px(size * 0.22)).overflow_hidden();
    match image {
        Some(i) => tile.bg(rgb(0xffffff)).child(img(i).size_full().object_fit(ObjectFit::Cover)),
        None => tile
            .bg(rgb(HOVER()))
            .flex()
            .items_center()
            .justify_center()
            .text_size(px(size * 0.42))
            .font_weight(FontWeight::BOLD)
            .text_color(rgb(MUTED()))
            .child(text.chars().next().map(|c| c.to_uppercase().to_string()).unwrap_or_default()),
    }
}

fn stat(icon_name: &'static str, text: String) -> gpui::Div {
    div()
        .h(px(26.))
        .px(px(9.))
        .flex()
        .items_center()
        .gap(px(6.))
        .rounded(px(6.))
        .bg(rgb(RAISED()))
        .border_1()
        .border_color(rgb(BORDER_SOFT()))
        .child(icon(icon_name).size(px(13.)).text_color(rgb(MUTED())))
        .child(div().text_size(px(12.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(text))
}

fn spec(key: &str, value: String) -> gpui::Div {
    div()
        .flex()
        .items_center()
        .gap(px(7.))
        .child(div().text_size(px(10.5)).text_color(rgb(MUTED_2())).child(key.to_uppercase()))
        .child(div().font_family(MONO).text_size(px(12.)).text_color(rgb(TEXT())).child(value))
}

fn badge(icon_name: &'static str, text: &str, color: u32) -> gpui::Div {
    div()
        .h(px(24.))
        .px(px(9.))
        .flex()
        .items_center()
        .gap(px(5.))
        .rounded_full()
        .bg(rgba((color << 8) | 0x1f))
        .border_1()
        .border_color(rgba((color << 8) | 0x55))
        .child(icon(icon_name).size(px(12.)).text_color(rgb(color)))
        .child(div().font_family(MONO).text_size(px(11.)).text_color(rgb(color)).child(text.to_string()))
}

fn capability_badges(f: &FamilyInfo) -> Vec<gpui::Div> {
    let mut out = vec![];
    if f.variants.iter().any(|v| v.2) {
        out.push(badge("zap", "Turbo", ACCENT()));
    }
    if !f.sizes.is_empty() {
        out.push(badge("grid-3x3", &f.sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" · "), MUTED()));
    }
    if f.alpha {
        out.push(badge("layers", tr!("Alfa nativo", "Native alpha"), CYAN()));
    }
    if f.capabilities.iter().any(|c| c == "walk_sheet") {
        out.push(badge("footprints", tr!("Hojas de caminar", "Walk sheets"), GREEN()));
    }
    out
}

impl Studio {
    pub(super) fn open_hub(&mut self, family: Option<String>, cx: &mut Context<Self>) {
        self.hub_sel = family.or_else(|| self.hub_sel.clone()).or_else(|| Some(self.model.family.clone()));
        self.hub_open = true;
        self.request_hf();
        cx.notify();
    }

    fn family_row(&self, i: usize, f: &FamilyInfo, cx: &mut Context<Self>) -> AnyElement {
        let sel = self.hub_sel.as_deref() == Some(f.id.as_str());
        let info = self.hf_info(&f.repo);
        let summary = info.as_ref().map(|(i, _)| hf::summary(&i.readme)).filter(|s| !s.is_empty()).unwrap_or_else(|| f.repo.clone());
        let rows = self.model.files_of(&f.id).to_vec();
        let ready = self.model.family_ready(&f.id);
        let (bytes, _) = missing_bytes(self, &variant_rows(&rows, false));
        let status = if !ready && bytes > 0 { gb(bytes) } else { String::new() };
        let (fg, sub) = if sel { (ACCENT_INK(), ACCENT_INK()) } else { (TEXT(), MUTED_2()) };
        let id = f.id.clone();
        div()
            .kb(("hub-fam", i))
            .p(px(10.))
            .flex()
            .items_center()
            .gap(px(11.))
            .rounded(px(10.))
            .cursor_pointer()
            .when(sel, |d| d.bg(rgb(ACCENT())))
            .when(!sel, |d| d.hover(|s| s.bg(rgb(RAISED()))))
            .child(avatar(info.and_then(|(_, a)| a), &f.label, 38.))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .flex()
                    .flex_col()
                    .gap(px(3.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .child(div().text_size(px(13.5)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(fg)).child(f.label.clone()))
                            .when(ready, |d| d.child(icon("circle-check-big").size(px(13.)).text_color(rgb(if sel { ACCENT_INK() } else { BLUE() })))),
                    )
                    .child(div().line_clamp(1).text_ellipsis().text_size(px(11.5)).text_color(rgb(sub)).opacity(if sel { 0.75 } else { 1. }).child(summary)),
            )
            .when(!status.is_empty(), |d| d.child(div().flex_none().font_family(MONO).text_size(px(10.5)).text_color(rgb(sub)).child(status)))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.hub_sel = Some(id.clone());
                cx.notify();
            }))
            .into_any_element()
    }

    fn file_row(&self, key: String, r: FileRow, family: &str, cx: &mut Context<Self>) -> AnyElement {
        let path = std::path::Path::new(&r.path);
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let size = r.remote.as_ref().and_then(|x| self.remote_size(x));
        let busy = self.downloads.progress().running;
        let downloadable = !r.found && r.remote.is_some() && !busy;
        let source = r
            .remote
            .as_ref()
            .map(|x| format!("huggingface.co/{}", x.repo))
            .unwrap_or_else(|| tr!("Sin descarga: cópialo a mano", "No download: copy it by hand").into());
        let (fam, one) = (family.to_string(), r.clone());
        div()
            .min_h(px(34.))
            .px(px(12.))
            .flex()
            .items_center()
            .gap(px(10.))
            .border_t_1()
            .border_color(rgb(BORDER_SOFT()))
            .child(icon(if r.found { "check" } else { "x" }).size(px(13.)).text_color(rgb(if r.found {
                BLUE()
            } else if r.turbo_only || r.precision.is_some() {
                MUTED_2()
            } else {
                DANGER()
            })))
            .child(div().w(px(160.)).flex_none().text_size(px(12.)).text_color(rgb(TEXT())).child({
                let mut l = super::settings_view::role_label(&r.role).to_string();
                if r.turbo_only {
                    l.push_str(" · Turbo");
                }
                if let Some(p) = &r.precision {
                    l.push_str(&format!(" · {p}"));
                }
                l
            }))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .text_ellipsis()
                    .whitespace_nowrap()
                    .font_family(MONO)
                    .text_size(px(11.))
                    .text_color(rgb(MUTED()))
                    .child(name),
            )
            .child(div().flex_none().font_family(MONO).text_size(px(11.)).text_color(rgb(MUTED_2())).child(size.map(gb).unwrap_or_default()))
            .child(div().w(px(30.)).flex_none().flex().justify_end().when(downloadable, |d| {
                d.child(
                    icon_btn(SharedString::from(key), "download")
                        .tooltip(tooltip(&source))
                        .on_click(cx.listener(move |this, _, _, cx| this.download_rows(&fam, vec![one.clone()], cx))),
                )
            }))
            .into_any_element()
    }

    fn variant_option(&self, f: &FamilyInfo, turbo: bool, rows: &[FileRow], cx: &mut Context<Self>) -> AnyElement {
        let mine = variant_rows(rows, turbo);
        let installed = mine.iter().all(|r| r.found);
        let (bytes, complete) = missing_bytes(self, &mine);
        let name = f.variants.iter().find(|v| v.2 == turbo).map(|v| v.1.clone()).unwrap_or_default();
        let base = f.variants.iter().find(|v| v.2 == turbo).map(|v| v.0.clone()).unwrap_or_default();
        let busy = self.downloads.progress().running;
        let fam = f.id.clone();
        let missing: Vec<FileRow> = mine.iter().filter(|r| !r.found && r.remote.is_some()).cloned().collect();
        div()
            .h(px(48.))
            .px(px(12.))
            .flex()
            .items_center()
            .gap(px(10.))
            .rounded(px(9.))
            .bg(rgb(INPUT_BG()))
            .border_1()
            .border_color(rgb(BORDER_SOFT()))
            .child(
                div()
                    .px(px(7.))
                    .h(px(20.))
                    .flex()
                    .items_center()
                    .rounded(px(5.))
                    .bg(rgb(if turbo { ACCENT() } else { RAISED() }))
                    .text_size(px(10.5))
                    .font_weight(FontWeight::BOLD)
                    .text_color(rgb(if turbo { ACCENT_INK() } else { TEXT() }))
                    .child(name),
            )
            .child(div().flex_1().min_w_0().flex().flex_col().gap(px(2.)).child(div().text_size(px(12.5)).text_color(rgb(TEXT())).child(base.clone())).child(
                div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(if installed {
                    tr!("Todo en disco", "Everything on disk").to_string()
                } else if complete {
                    tr!(format!("{} archivos por descargar", missing.len()), format!("{} files to download", missing.len()))
                } else {
                    tr!("Algún archivo no tiene descarga", "Some files have no download").to_string()
                }),
            ))
            .when(!installed && bytes > 0, |d| d.child(div().font_family(MONO).text_size(px(11.5)).text_color(rgb(MUTED())).child(gb(bytes))))
            .when(installed, |d| {
                d.child(button_i(SharedString::from(format!("hub-use-{}-{turbo}", f.id)), "play", tr!("Usar", "Use")).on_click(cx.listener(
                    move |this, _, _, cx| {
                        this.send(json!({"cmd": "select", "base": base, "lora": "", "load": true}));
                        this.hub_open = false;
                        cx.notify();
                    },
                )))
            })
            .when(!installed && !missing.is_empty() && !busy, |d| {
                d.child(
                    div()
                        .kb(SharedString::from(format!("hub-dl-{}-{turbo}", f.id)))
                        .h(px(30.))
                        .px(px(12.))
                        .flex()
                        .items_center()
                        .gap(px(7.))
                        .rounded(px(7.))
                        .bg(rgb(ACCENT()))
                        .cursor_pointer()
                        .hover(|s| s.bg(rgb(ACCENT_HOVER())))
                        .child(icon("download").size(px(14.)).text_color(rgb(ACCENT_INK())))
                        .child(div().text_size(px(12.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(ACCENT_INK())).child(tr!("Descargar", "Download")))
                        .on_click(cx.listener(move |this, _, _, cx| this.download_rows(&fam, missing.clone(), cx))),
                )
            })
            .into_any_element()
    }

    fn family_detail(&self, f: &FamilyInfo, cx: &mut Context<Self>) -> AnyElement {
        let state = self.hf.get(&f.repo);
        let info = match &state {
            Some(State::Ready(i, a)) => Some((i.clone(), a.clone())),
            _ => None,
        };
        let rows = self.model.files_of(&f.id).to_vec();
        let repo = f.repo.clone();
        let mut header = div()
            .flex()
            .items_center()
            .gap(px(12.))
            .child(avatar(info.as_ref().and_then(|(_, a)| a.clone()), &f.label, 44.))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(div().text_size(px(18.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(f.label.clone()))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .child(div().font_family(MONO).text_size(px(11.5)).text_color(rgb(MUTED())).child(repo.clone()))
                            .child(
                                icon_btn("hub-copy", "copy").w(px(22.)).h(px(22.)).tooltip(tooltip(tr!("Copiar el id del repo", "Copy the repo id"))).on_click(
                                    cx.listener({
                                        let r = repo.clone();
                                        move |_, _, _, cx| cx.write_to_clipboard(ClipboardItem::new_string(r.clone()))
                                    }),
                                ),
                            ),
                    ),
            )
            .child(button_i("hub-open-hf", "external-link", "Hugging Face").on_click(cx.listener({
                let r = repo.clone();
                move |_, _, _, cx| cx.open_url(&format!("https://huggingface.co/{r}"))
            })));
        header = header.child(icon_btn("hub-close", "x").on_click(cx.listener(|this, _, _, cx| {
            this.hub_open = false;
            cx.notify();
        })));

        let stats = div().flex().flex_wrap().items_center().gap(px(8.)).children(
            info.as_ref()
                .map(|(i, _)| {
                    let mut v = vec![stat("download", hf::count(i.downloads)), stat("star", hf::count(i.likes))];
                    if !i.last_modified.is_empty() {
                        v.push(
                            div()
                                .text_size(px(12.))
                                .text_color(rgb(MUTED()))
                                .child(tr!(format!("Actualizado {}", hf::age(&i.last_modified)), format!("Updated {}", hf::age(&i.last_modified)))),
                        );
                    }
                    v
                })
                .unwrap_or_default(),
        );

        let card_text: AnyElement = match (&state, &info) {
            (_, Some((i, _))) if !hf::summary(&i.readme).is_empty() => {
                div().text_size(px(13.5)).line_height(px(21.)).text_color(rgb(ACCENT())).child(hf::summary(&i.readme)).into_any_element()
            }
            (_, Some((i, _))) if i.gated => div()
                .text_size(px(12.5))
                .line_height(px(19.))
                .text_color(rgb(MUTED()))
                .flex()
                .items_center()
                .gap(px(12.))
                .child(div().flex_1().child(tr!(
                    "Este repo pide aceptar su licencia en Hugging Face (gratis). Con tu token añadido se ven la ficha y la descarga.",
                    "This repo asks you to accept its license on Hugging Face (free). With your token added you get the card and the download."
                )))
                .child(button_i("hub-token", "key-round", tr!("Configurar token", "Set up token")).on_click(cx.listener(|this, _, _, cx| {
                    this.hub_open = false;
                    this.settings_tab = super::settings_view::Tab::HuggingFace;
                    this.settings_open = true;
                    cx.notify();
                })))
                .into_any_element(),
            (Some(State::Failed(e)), _) => div()
                .text_size(px(12.))
                .text_color(rgb(MUTED()))
                .child(tr!(format!("Sin conexión con Hugging Face: {e}"), format!("Could not reach Hugging Face: {e}")))
                .into_any_element(),
            _ => div()
                .flex()
                .items_center()
                .gap(px(8.))
                .child(crate::icons::spinner("hub-spin", 13., MUTED_2()))
                .child(div().text_size(px(12.)).text_color(rgb(MUTED_2())).child(tr!("Consultando Hugging Face…", "Asking Hugging Face…")))
                .into_any_element(),
        };
        let mut specs = vec![spec(tr!("arq", "arch"), f.id.clone()), spec(tr!("salida", "output"), traits(f).join(" · "))];
        if let Some((i, _)) = &info {
            if i.parameters > 0 {
                specs.insert(0, spec(tr!("parámetros", "params"), format!("{:.1}B", i.parameters as f64 / 1e9)));
            }
            if !i.task.is_empty() {
                specs.push(spec(tr!("tarea", "task"), i.task.clone()));
            }
            if !i.license.is_empty() {
                specs.push(spec(tr!("licencia", "license"), i.license.clone()));
            }
        }
        let summary_card = div()
            .p(px(16.))
            .rounded(px(12.))
            .bg(rgb(RAISED()))
            .border_1()
            .border_color(rgb(BORDER_SOFT()))
            .flex()
            .flex_col()
            .gap(px(14.))
            .child(card_text)
            .child(div().h(px(1.)).bg(rgb(BORDER_SOFT())))
            .child(div().flex().flex_wrap().gap_x(px(18.)).gap_y(px(8.)).children(specs))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(8.))
                    .child(div().text_size(px(10.5)).text_color(rgb(MUTED_2())).child(tr!("CAPACIDADES", "CAPABILITIES")))
                    .children(capability_badges(f)),
            );

        let files: Vec<AnyElement> = rows.iter().cloned().enumerate().map(|(i, r)| self.file_row(format!("hub-file-{}-{i}", f.id), r, &f.id, cx)).collect();
        let options = div()
            .flex()
            .flex_col()
            .gap(px(8.))
            .child(self.variant_option(f, false, &rows, cx))
            .when(f.variants.iter().any(|v| v.2), |d| d.child(self.variant_option(f, true, &rows, cx)))
            .children(self.download_progress(&f.id, cx))
            .child(
                div()
                    .mt(px(4.))
                    .rounded(px(9.))
                    .border_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .overflow_hidden()
                    .child(div().h(px(30.)).px(px(12.)).flex().items_center().bg(rgb(BG())).child(label(tr!("Archivos", "Files"))))
                    .children(files),
            );

        let readme: Vec<AnyElement> = info
            .as_ref()
            .map(|(i, _)| hf::blocks(&i.readme))
            .unwrap_or_default()
            .into_iter()
            .take(40)
            .map(|b| match b {
                Block::Heading(h) => div().pt(px(10.)).text_size(px(15.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(h).into_any_element(),
                Block::Paragraph(p) => div().text_size(px(13.)).line_height(px(20.)).text_color(rgb(MUTED())).child(p).into_any_element(),
                Block::Bullet(t) => div()
                    .flex()
                    .gap(px(8.))
                    .child(div().text_color(rgb(MUTED_2())).child("•"))
                    .child(div().flex_1().text_size(px(13.)).line_height(px(20.)).text_color(rgb(MUTED())).child(t))
                    .into_any_element(),
            })
            .collect();

        div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(header)
            .child(stats)
            .child(summary_card)
            .child(
                div()
                    .pt(px(4.))
                    .text_size(px(14.))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(TEXT()))
                    .child(tr!("Opciones de descarga", "Download options")),
            )
            .child(options)
            .when(!readme.is_empty(), |d| {
                d.child(div().pt(px(4.)).text_size(px(14.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child("README")).child(
                    div()
                        .p(px(16.))
                        .rounded(px(12.))
                        .bg(rgb(RAISED()))
                        .border_1()
                        .border_color(rgb(BORDER_SOFT()))
                        .flex()
                        .flex_col()
                        .gap(px(10.))
                        .children(readme),
                )
            })
            .into_any_element()
    }

    fn store_row(&self, i: usize, repo: &str, name: &str, cx: &mut Context<Self>) -> AnyElement {
        let key = format!("store:{repo}");
        let sel = self.hub_sel.as_deref() == Some(key.as_str());
        let info = self.hf_info(repo);
        let summary = info.as_ref().map(|(i, _)| hf::summary(&i.readme)).filter(|s| !s.is_empty()).unwrap_or_else(|| repo.to_string());
        let (fg, sub) = if sel { (ACCENT_INK(), ACCENT_INK()) } else { (TEXT(), MUTED_2()) };
        div()
            .kb(("hub-store", i))
            .p(px(10.))
            .flex()
            .items_center()
            .gap(px(11.))
            .rounded(px(10.))
            .cursor_pointer()
            .when(sel, |d| d.bg(rgb(ACCENT())))
            .when(!sel, |d| d.hover(|s| s.bg(rgb(RAISED()))))
            .child(div().opacity(if sel { 1. } else { 0.7 }).child(avatar(info.and_then(|(_, a)| a), name, 38.)))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .flex()
                    .flex_col()
                    .gap(px(3.))
                    .child(div().text_size(px(13.5)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(fg)).child(name.to_string()))
                    .child(div().line_clamp(1).text_ellipsis().text_size(px(11.5)).text_color(rgb(sub)).opacity(if sel { 0.75 } else { 1. }).child(summary)),
            )
            .on_click(cx.listener(move |this, _, _, cx| {
                this.hub_sel = Some(key.clone());
                cx.notify();
            }))
            .into_any_element()
    }

    /// A model listed before Dotsmith can run it: its Hugging Face card and why there is nothing to download yet.
    fn store_detail(&self, repo: &str, name: &str, cx: &mut Context<Self>) -> AnyElement {
        let state = self.hf.get(repo);
        let info = match &state {
            Some(State::Ready(i, a)) => Some((i.clone(), a.clone())),
            _ => None,
        };
        let r = repo.to_string();
        let header = div()
            .flex()
            .items_center()
            .gap(px(12.))
            .child(avatar(info.as_ref().and_then(|(_, a)| a.clone()), name, 44.))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(div().text_size(px(18.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(name.to_string()))
                            .child(
                                div()
                                    .h(px(20.))
                                    .px(px(8.))
                                    .flex()
                                    .items_center()
                                    .rounded_full()
                                    .bg(rgb(RAISED()))
                                    .border_1()
                                    .border_color(rgb(BORDER()))
                                    .text_size(px(10.5))
                                    .text_color(rgb(MUTED()))
                                    .child(tr!("Próximamente", "Coming soon")),
                            ),
                    )
                    .child(div().font_family(MONO).text_size(px(11.5)).text_color(rgb(MUTED())).child(repo.to_string())),
            )
            .child(
                button_i("hub-open-hf", "external-link", "Hugging Face")
                    .on_click(cx.listener(move |_, _, _, cx| cx.open_url(&format!("https://huggingface.co/{r}")))),
            )
            .child(icon_btn("hub-close", "x").on_click(cx.listener(|this, _, _, cx| {
                this.hub_open = false;
                cx.notify();
            })));
        let stats = div().flex().flex_wrap().items_center().gap(px(8.)).children(
            info.as_ref()
                .map(|(i, _)| {
                    let mut v = vec![stat("download", hf::count(i.downloads)), stat("star", hf::count(i.likes))];
                    if !i.license.is_empty() {
                        v.push(div().text_size(px(12.)).text_color(rgb(MUTED())).child(format!("{} {}", tr!("Licencia", "License"), i.license)));
                    }
                    v
                })
                .unwrap_or_default(),
        );
        let text: AnyElement = match (&state, &info) {
            (_, Some((i, _))) if !hf::summary(&i.readme).is_empty() => {
                div().text_size(px(13.5)).line_height(px(21.)).text_color(rgb(ACCENT())).child(hf::summary(&i.readme)).into_any_element()
            }
            (Some(State::Failed(e)), _) => div()
                .text_size(px(12.))
                .text_color(rgb(MUTED()))
                .child(tr!(format!("Sin conexión con Hugging Face: {e}"), format!("Could not reach Hugging Face: {e}")))
                .into_any_element(),
            _ => div()
                .flex()
                .items_center()
                .gap(px(8.))
                .child(crate::icons::spinner("hub-spin", 13., MUTED_2()))
                .child(div().text_size(px(12.)).text_color(rgb(MUTED_2())).child(tr!("Consultando Hugging Face…", "Asking Hugging Face…")))
                .into_any_element(),
        };
        let readme: Vec<AnyElement> = info
            .as_ref()
            .map(|(i, _)| hf::blocks(&i.readme))
            .unwrap_or_default()
            .into_iter()
            .take(40)
            .map(|b| match b {
                Block::Heading(h) => div().pt(px(10.)).text_size(px(15.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(h).into_any_element(),
                Block::Paragraph(p) => div().text_size(px(13.)).line_height(px(20.)).text_color(rgb(MUTED())).child(p).into_any_element(),
                Block::Bullet(t) => div()
                    .flex()
                    .gap(px(8.))
                    .child(div().text_color(rgb(MUTED_2())).child("•"))
                    .child(div().flex_1().text_size(px(13.)).line_height(px(20.)).text_color(rgb(MUTED())).child(t))
                    .into_any_element(),
            })
            .collect();
        div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(header)
            .child(stats)
            .child(div().p(px(16.)).rounded(px(12.)).bg(rgb(RAISED())).border_1().border_color(rgb(BORDER_SOFT())).child(text))
            .child(
                div()
                    .p(px(14.))
                    .flex()
                    .items_start()
                    .gap(px(10.))
                    .rounded(px(10.))
                    .bg(rgb(BG()))
                    .border_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .child(icon("info").size(px(16.)).text_color(rgb(MUTED())))
                    .child(div().flex_1().text_size(px(12.5)).line_height(px(18.)).text_color(rgb(MUTED())).child(tr!(
                        "Dotsmith todavía no ejecuta esta arquitectura. Cuando lo haga, aparecerá aquí lista para descargar y usar.",
                        "Dotsmith can't run this architecture yet. When it can, it will be here ready to download and use."
                    ))),
            )
            .when(!readme.is_empty(), |d| {
                d.child(div().pt(px(4.)).text_size(px(14.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child("README")).child(
                    div()
                        .p(px(16.))
                        .rounded(px(12.))
                        .bg(rgb(RAISED()))
                        .border_1()
                        .border_color(rgb(BORDER_SOFT()))
                        .flex()
                        .flex_col()
                        .gap(px(10.))
                        .children(readme),
                )
            })
            .into_any_element()
    }

    pub(super) fn hub_view(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let query = self.hub_query.read(cx).text().to_lowercase();
        let families: Vec<FamilyInfo> = self
            .model
            .families
            .iter()
            .filter(|f| query.is_empty() || f.label.to_lowercase().contains(&query) || f.repo.to_lowercase().contains(&query))
            .cloned()
            .collect();
        if self.hub_sel.is_none() && !self.model.family.is_empty() {
            self.hub_sel = Some(self.model.family.clone());
        }
        let selected = self.model.families.iter().find(|f| Some(&f.id) == self.hub_sel.as_ref()).cloned();
        let mut rows: Vec<AnyElement> = families.iter().enumerate().map(|(i, f)| self.family_row(i, f, cx)).collect();
        let store: Vec<(String, String)> = self
            .model
            .store
            .iter()
            .filter(|(repo, label)| query.is_empty() || label.to_lowercase().contains(&query) || repo.to_lowercase().contains(&query))
            .cloned()
            .collect();
        if !store.is_empty() {
            rows.push(div().px(px(10.)).pt(px(14.)).pb(px(4.)).child(label(tr!("Próximamente", "Coming soon"))).into_any_element());
            rows.extend(store.iter().enumerate().map(|(i, (repo, name))| self.store_row(i, repo, name, cx)));
        }
        let store_sel = self.hub_sel.as_deref().and_then(|s| s.strip_prefix("store:")).and_then(|r| self.model.store.iter().find(|s| s.0 == r)).cloned();
        let detail = match store_sel {
            Some((repo, name)) => Some(self.store_detail(&repo, &name, cx)),
            None => selected.map(|f| self.family_detail(&f, cx)),
        };
        let card = div()
            .id("hub")
            .occlude()
            .w(px(1120.))
            .h(px(740.))
            .max_w_full()
            .max_h_full()
            .rounded(px(14.))
            .bg(rgb(PANEL()))
            .border_1()
            .border_color(rgb(BORDER()))
            .shadow_lg()
            .flex()
            .overflow_hidden()
            .child(
                div()
                    .w(px(380.))
                    .flex_none()
                    .flex()
                    .flex_col()
                    .bg(rgb(BG()))
                    .border_r_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .child(
                        div()
                            .p(px(12.))
                            .flex()
                            .flex_col()
                            .gap(px(10.))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .px(px(4.))
                                    .child(icon("box").size(px(16.)).text_color(rgb(ACCENT())))
                                    .child(
                                        div().flex_1().text_size(px(15.)).font_weight(FontWeight::BOLD).text_color(rgb(TEXT())).child(tr!("Modelos", "Models")),
                                    )
                                    .child(icon_btn("hub-refresh", "rotate-ccw").tooltip(tooltip(tr!("Actualizar", "Refresh"))).on_click(cx.listener(
                                        |this, _, _, cx| {
                                            this.hf.refresh();
                                            this.request_hf();
                                            this.send(json!({"cmd": "state"}));
                                            cx.notify();
                                        },
                                    ))),
                            )
                            .child(self.hub_query.clone()),
                    )
                    .child(div().id("hub-list").flex_1().min_h_0().overflow_y_scroll().px(px(8.)).pb(px(8.)).flex().flex_col().gap(px(2.)).children(rows)),
            )
            .child(div().id("hub-detail").flex_1().min_w_0().overflow_y_scroll().p(px(22.)).children(detail));
        self.modal_layer("hub-veil", card, |s, _, _| s.hub_open = false, cx)
    }

    /// The downloads panel: the running batch with per-file progress, then what finished in this session.
    pub(super) fn downloads_view(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let p = self.downloads.progress();
        let active: Vec<AnyElement> = if p.running {
            p.items
                .iter()
                .enumerate()
                .map(|(i, (name, size))| {
                    let (done, state) = if i < p.index {
                        (*size, 2)
                    } else if i == p.index {
                        (p.done, 1)
                    } else {
                        (0, 0)
                    };
                    let frac = (*size > 0).then(|| done as f32 / *size as f32);
                    div()
                        .px(px(14.))
                        .py(px(10.))
                        .flex()
                        .items_center()
                        .gap(px(10.))
                        .border_b_1()
                        .border_color(rgb(BORDER_SOFT()))
                        .child(icon(if state == 2 { "check" } else { "download" }).size(px(15.)).text_color(rgb(if state == 1 { ACCENT() } else { MUTED_2() })))
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .flex()
                                .flex_col()
                                .gap(px(4.))
                                .child(
                                    div()
                                        .min_w_0()
                                        .overflow_hidden()
                                        .text_ellipsis()
                                        .whitespace_nowrap()
                                        .text_size(px(12.))
                                        .text_color(rgb(TEXT()))
                                        .child(name.clone()),
                                )
                                .child(if state == 1 {
                                    div().flex().items_center().gap(px(8.)).child(super::model_picker::progress_bar(frac, 200.)).child(
                                        div().font_family(MONO).text_size(px(10.5)).text_color(rgb(MUTED())).child(format!("{} / {}", gb(done), gb(*size))),
                                    )
                                } else {
                                    div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(if state == 2 {
                                        gb(*size)
                                    } else {
                                        format!("{} · {}", tr!("En cola", "Queued"), gb(*size))
                                    })
                                }),
                        )
                        .into_any_element()
                })
                .collect()
        } else {
            vec![]
        };
        let finished: Vec<AnyElement> = p
            .history
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let dest = f.dest.clone();
                div()
                    .px(px(14.))
                    .py(px(10.))
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .border_b_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .child(icon("box").size(px(15.)).text_color(rgb(MUTED())))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(
                                div()
                                    .min_w_0()
                                    .overflow_hidden()
                                    .text_ellipsis()
                                    .whitespace_nowrap()
                                    .text_size(px(12.))
                                    .text_color(rgb(TEXT()))
                                    .child(f.name.clone()),
                            )
                            .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(gb(f.bytes))),
                    )
                    .child(
                        icon_btn(("dl-reveal", i), "folder-open")
                            .tooltip(tooltip(tr!("Mostrar en la carpeta", "Show in folder")))
                            .on_click(cx.listener(move |_, _, _, _| super::reveal_in_explorer(&dest))),
                    )
                    .into_any_element()
            })
            .collect();
        let empty = active.is_empty() && finished.is_empty();
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .child(div().id("dl-backdrop").absolute().top_0().left_0().size_full().on_click(cx.listener(|this, _, _, cx| {
                this.downloads_open = false;
                cx.notify();
            })))
            .child(
                div()
                    .id("downloads")
                    .occlude()
                    .absolute()
                    .top(px(40.))
                    .right(px(181.))
                    .w(px(400.))
                    .max_h(px(520.))
                    .rounded(px(12.))
                    .bg(rgb(PANEL()))
                    .border_1()
                    .border_color(rgb(BORDER()))
                    .shadow_lg()
                    .flex()
                    .flex_col()
                    .overflow_hidden()
                    .child(
                        div()
                            .px(px(14.))
                            .py(px(12.))
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .border_b_1()
                            .border_color(rgb(BORDER_SOFT()))
                            .child(
                                div()
                                    .flex_1()
                                    .text_size(px(14.))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(rgb(TEXT()))
                                    .child(tr!("Descargas", "Downloads")),
                            )
                            .when(p.running, |d| {
                                d.child(button("dl-cancel-all", tr!("Cancelar", "Cancel")).on_click(cx.listener(|this, _, _, cx| {
                                    this.downloads.cancel();
                                    cx.notify();
                                })))
                            })
                            .child(icon_btn("dl-close", "x").on_click(cx.listener(|this, _, _, cx| {
                                this.downloads_open = false;
                                cx.notify();
                            }))),
                    )
                    .child(
                        div()
                            .id("dl-list")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .children(active)
                            .when(!finished.is_empty(), |d| d.child(div().px(px(14.)).pt(px(10.)).pb(px(4.)).child(label(tr!("Completadas", "Completed")))))
                            .children(finished)
                            .when(empty, |d| {
                                d.child(
                                    div()
                                        .p(px(24.))
                                        .flex()
                                        .flex_col()
                                        .items_center()
                                        .gap(px(8.))
                                        .child(icon("download").size(px(22.)).text_color(rgb(MUTED_2())))
                                        .child(div().text_size(px(12.5)).text_color(rgb(MUTED())).child(tr!("No hay descargas.", "No downloads."))),
                                )
                            }),
                    )
                    .child(
                        div()
                            .id("dl-open-dir")
                            .px(px(14.))
                            .py(px(10.))
                            .flex()
                            .justify_end()
                            .border_t_1()
                            .border_color(rgb(BORDER_SOFT()))
                            .cursor_pointer()
                            .text_size(px(11.5))
                            .text_color(rgb(MUTED()))
                            .hover(|s| s.text_color(rgb(TEXT())))
                            .child(tr!("Abrir la carpeta de modelos", "Open the models folder"))
                            .on_click(cx.listener(|_, _, _, cx| {
                                if let Some(dir) = dotsmith_engine::locations::download_root() {
                                    let _ = std::fs::create_dir_all(&dir);
                                    cx.open_with_system(&dir);
                                }
                            })),
                    ),
            )
            .into_any_element()
    }
}

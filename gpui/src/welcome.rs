//! The start screen (Obsidian's vault picker in spirit): a window of its own before the studio, shown while Dotsmith
//! cannot generate yet. One row per thing to sort out, each with its action on the right; "Start creating" opens the
//! studio once a base model is ready.
use gpui::{AnyElement, Context, FontWeight, div, prelude::*, px, rgb, rgba};

use super::{FileRow, Keyboard, Studio, button, seg, segmented, setting_row, settings_view};
use crate::i18n::Lang;
use crate::icons::icon;
use crate::theme::*;
use crate::tr;

fn done_mark() -> gpui::Div {
    div()
        .w(px(26.))
        .h(px(26.))
        .rounded_full()
        .bg(rgba(0xe5a95f26))
        .flex()
        .items_center()
        .justify_center()
        .child(icon("check").size(px(14.)).text_color(rgb(BLUE())))
}

impl Studio {
    /// Whether the start screen covers the studio: from the first state report until the user starts, and whenever
    /// the GPU or the base model go missing later.
    pub(super) fn welcome_visible(&self) -> bool {
        let Some(gpu) = &self.model.gpu else { return false };
        self.onboarding == Some(true) || !gpu.ok || !self.model.can_generate()
    }

    pub(super) fn welcome_view(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let gpu = self.model.gpu.clone().unwrap_or_default();
        let fam = self.model.families.iter().find(|f| f.id == self.model.family).cloned().unwrap_or_default();
        let ready = gpu.ok && self.model.can_generate();
        let busy = self.downloads.progress().running;
        let loras: usize = self.model.runs.iter().map(|r| r.ckpts.len()).sum();
        let chosen = self.settings.language.clone();

        let gpu_row = setting_row(
            tr!("Tarjeta gráfica", "Graphics card"),
            if gpu.ok { format!("{} · {:.0} GB", gpu.name, gpu.vram_mb as f64 / 1024.) } else { gpu.error.clone() },
            if gpu.ok {
                done_mark().into_any_element()
            } else {
                button("w-gpu", tr!("Detalles", "Details"))
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.settings_tab = settings_view::Tab::Gpu;
                        this.settings_open = true;
                        cx.notify();
                    }))
                    .into_any_element()
            },
        );

        let families = self.model.families.clone();
        let current = self.model.family.clone();
        let model_rows: Vec<AnyElement> = families
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let installed = self.model.family_ready(&f.id);
                let fits = self.model.runs_here(f);
                let rows: Vec<FileRow> = self.model.files_of(&f.id).iter().filter(|r| !r.found && !r.turbo_only && r.precision.is_none()).cloned().collect();
                let bytes: u64 = rows.iter().filter_map(|r| r.remote.as_ref().and_then(|x| self.remote_size(x))).sum();
                let can_download = !rows.is_empty() && rows.iter().all(|r| r.remote.is_some());
                let info = self.hf_info(&f.repo);
                let summary = info.as_ref().map(|(i, _)| crate::hf::summary(&i.readme)).filter(|s| !s.is_empty());
                let mut facts = vec![self.publisher(f)];
                if let Some((i, _)) = &info
                    && i.parameters > 0
                {
                    facts.push(format!("{:.1}B", i.parameters as f64 / 1e9));
                }
                if !installed && bytes > 0 {
                    facts.push(crate::download::gb(bytes));
                }
                let base = f.variants.first().map(|v| v.0.clone()).unwrap_or_default();
                let (id, id2) = (f.id.clone(), f.id.clone());
                let action: AnyElement = if !fits {
                    div().text_size(px(11.5)).text_color(rgb(MUTED_2())).child(super::ModelState::needs_newer(f)).into_any_element()
                } else if installed && f.id == current {
                    div()
                        .flex()
                        .items_center()
                        .gap(px(6.))
                        .child(div().text_size(px(11.5)).text_color(rgb(BLUE())).child(tr!("En uso", "In use")))
                        .child(done_mark())
                        .into_any_element()
                } else if installed {
                    button(("w-use", i), tr!("Usar", "Use"))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.send(serde_json::json!({"cmd": "select", "base": base, "lora": "-"}));
                            cx.notify();
                        }))
                        .into_any_element()
                } else if can_download && !busy {
                    div()
                        .kb(("w-dl", i))
                        .h(px(32.))
                        .px(px(14.))
                        .flex()
                        .items_center()
                        .gap(px(7.))
                        .rounded(px(8.))
                        .bg(rgb(RAISED()))
                        .border_1()
                        .border_color(rgb(BORDER()))
                        .cursor_pointer()
                        .hover(|s| s.bg(rgb(HOVER())).border_color(rgb(ACCENT_DIM())))
                        .child(icon("download").size(px(14.)).text_color(rgb(TEXT())))
                        .child(div().text_size(px(12.5)).text_color(rgb(TEXT())).child(tr!("Descargar", "Download")))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.send(serde_json::json!({"cmd": "select", "base": base, "lora": "-"}));
                            this.download_rows(&id, rows.clone(), cx);
                        }))
                        .into_any_element()
                } else {
                    div().into_any_element()
                };
                let progress = self.download_progress(&f.id, cx);
                div()
                    .flex()
                    .flex_col()
                    .py(px(12.))
                    .border_t_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(12.))
                            .child(super::model_hub::avatar(info.as_ref().and_then(|(_, a)| a.clone()), &f.label, 36.))
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .flex()
                                    .flex_col()
                                    .gap(px(3.))
                                    .child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .gap(px(8.))
                                            .child(div().text_size(px(13.5)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(f.label.clone()))
                                            .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(facts.join(" · "))),
                                    )
                                    .children(summary.map(|s| div().line_clamp(1).text_ellipsis().text_size(px(11.5)).text_color(rgb(MUTED())).child(s))),
                            )
                            .child(
                                div()
                                    .kb(("w-card", i))
                                    .text_size(px(11.5))
                                    .text_color(rgb(MUTED_2()))
                                    .cursor_pointer()
                                    .hover(|s| s.text_color(rgb(TEXT())))
                                    .child(tr!("Ficha", "Details"))
                                    .on_click(cx.listener(move |this, _, _, cx| this.open_hub(Some(id2.clone()), cx))),
                            )
                            .child(action),
                    )
                    .children(progress.map(|p| div().pl(px(48.)).pt(px(8.)).child(p)))
                    .into_any_element()
            })
            .collect();

        let have_row = (!self.model.family_ready(&fam.id)).then(|| {
            setting_row(
                tr!("Ya tengo los modelos", "I already have the models"),
                tr!("Elige la carpeta donde los guardas.", "Choose the folder where you keep them."),
                button("w-have", tr!("Elegir carpeta…", "Choose folder…"))
                    .on_click(cx.listener(|this, _, _, cx| this.pick_folder(cx, |s, p| s.models_dir = Some(p)))),
            )
        });

        let lora_row = setting_row(
            tr!("LoRAs · opcional", "LoRAs · optional"),
            if loras > 0 {
                tr!(format!("{loras} en tu biblioteca"), format!("{loras} in your library"))
            } else {
                tr!("Copia archivos .safetensors en su carpeta.", "Copy .safetensors files into their folder.").into()
            },
            button("w-loras", tr!("Abrir carpeta", "Open folder")).on_click(cx.listener(|this, _, _, cx| {
                if let Some(dir) = this.settings.lora_library() {
                    let _ = std::fs::create_dir_all(&dir);
                    cx.open_with_system(&dir);
                }
            })),
        );

        let start = div()
            .kb("w-start")
            .mt(px(28.))
            .h(px(44.))
            .w_full()
            .rounded(px(10.))
            .flex()
            .items_center()
            .justify_center()
            .gap(px(8.))
            .text_size(px(14.))
            .font_weight(FontWeight::BOLD)
            .when(ready, |d| {
                d.bg(rgb(ACCENT())).text_color(rgb(ACCENT_INK())).cursor_pointer().hover(|s| s.bg(rgb(ACCENT_HOVER()))).on_click(cx.listener(
                    |this, _, _, cx| {
                        this.onboarding = Some(false);
                        cx.notify();
                    },
                ))
            })
            .when(!ready, |d| d.bg(rgb(RAISED())).border_1().border_color(rgb(BORDER())).text_color(rgb(MUTED_2())))
            .child(if ready { tr!("Empezar a crear", "Start creating") } else { tr!("Elige un modelo base para empezar", "Choose a base model to start") });

        let langs = Lang::ALL
            .into_iter()
            .enumerate()
            .map(|(i, l)| {
                seg(("w-lang", i), l.native(), chosen.as_deref() == Some(l.code()))
                    .on_click(cx.listener(move |this, _, _, cx| this.update_settings(|s| s.language = Some(l.code().into()), cx)))
            })
            .chain(std::iter::once(
                seg("w-lang-sys", tr!("Sistema", "System"), chosen.is_none())
                    .on_click(cx.listener(|this, _, _, cx| this.update_settings(|s| s.language = None, cx))),
            ));

        div()
            .id("welcome")
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .bg(rgb(BG()))
            .flex()
            .flex_col()
            .items_center()
            .child(
                div()
                    .w(px(560.))
                    .max_w_full()
                    .pt(px(56.))
                    .pb(px(32.))
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .items_center()
                            .gap(px(6.))
                            .pb(px(30.))
                            .child(gpui::img(self.logo.clone()).w(px(72.)).h(px(72.)))
                            .child(div().text_size(px(26.)).font_weight(FontWeight::BOLD).text_color(rgb(TEXT())).child("Dotsmith"))
                            .child(div().text_size(px(12.)).text_color(rgb(MUTED_2())).child(format!(
                                "{} {}",
                                tr!("Versión", "Version"),
                                env!("CARGO_PKG_VERSION")
                            ))),
                    )
                    .child(gpu_row)
                    .child(
                        div()
                            .pt(px(22.))
                            .pb(px(4.))
                            .text_size(px(11.))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(rgb(MUTED_2()))
                            .child(tr!("ELIGE UN MODELO BASE", "CHOOSE A BASE MODEL")),
                    )
                    .children(model_rows)
                    .children(have_row)
                    .child(div().pt(px(18.)))
                    .child(lora_row.border_b_1())
                    .child(start)
                    .child(div().pt(px(36.)).flex().justify_center().child(segmented(langs))),
            )
            .into_any_element()
    }
}

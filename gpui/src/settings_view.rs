//! The Settings window: a page list on the left, one setting per row on the right (Obsidian's layout). Every
//! change is saved at once; there is no Apply.
use gpui::{AnyElement, Context, FontWeight, PathPromptOptions, SharedString, div, prelude::*, px, rgb, rgba};
use serde_json::json;
use std::path::{Path, PathBuf};

use super::{FileRow, Keyboard, Level, Studio, button, button_i, folders, icon_btn, kbd, seg, segmented, setting_row, tooltip};
use crate::download::{Item, gb};
use crate::i18n::{self, Lang};
use crate::icons::icon;
use crate::settings::{Settings, display_path};
use crate::theme::*;
use crate::tr;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Tab {
    #[default]
    General,
    Models,
    Gpu,
    Shortcuts,
    About,
}

impl Tab {
    const ALL: [Tab; 5] = [Tab::General, Tab::Models, Tab::Gpu, Tab::Shortcuts, Tab::About];

    pub fn from_key(k: &str) -> Option<Tab> {
        match k {
            "general" => Some(Tab::General),
            "models" => Some(Tab::Models),
            "gpu" => Some(Tab::Gpu),
            "shortcuts" => Some(Tab::Shortcuts),
            "about" => Some(Tab::About),
            _ => None,
        }
    }

    pub(super) fn title(self) -> &'static str {
        match self {
            Tab::General => tr!("General", "General"),
            Tab::Models => tr!("Modelos y LoRAs", "Models & LoRAs"),
            Tab::Gpu => tr!("GPU y memoria", "GPU & memory"),
            Tab::Shortcuts => tr!("Atajos", "Shortcuts"),
            Tab::About => tr!("Acerca de", "About"),
        }
    }

    pub(super) fn glyph(self) -> &'static str {
        match self {
            Tab::General => "settings",
            Tab::Models => "box",
            Tab::Gpu => "cpu",
            Tab::Shortcuts => "keyboard",
            Tab::About => "info",
        }
    }
}

/// Progress label of downloads started from a link (families use their id).
const LINK: &str = "link";

/// "owner/repo/path", "owner/repo/blob/<rev>/path" or a huggingface.co URL as (repo, revision, path).
fn parse_hf_link(text: &str) -> Option<(String, String, String)> {
    let t = text.trim().trim_start_matches("https://").trim_start_matches("http://").trim_start_matches("huggingface.co/");
    let parts: Vec<&str> = t.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 3 {
        return None;
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    match parts[2] {
        "resolve" | "blob" if parts.len() >= 5 => Some((repo, parts[3].to_string(), parts[4..].join("/"))),
        _ => Some((repo, "main".to_string(), parts[2..].join("/"))),
    }
}

pub(super) fn role_label(role: &str) -> &'static str {
    match role {
        "diffusion model" => tr!("Modelo de difusión", "Diffusion model"),
        "text encoder" => tr!("Codificador de texto", "Text encoder"),
        "VAE" => "VAE",
        _ => "Tokenizer",
    }
}

fn group(title: &str) -> gpui::Div {
    div().pt(px(22.)).pb(px(4.)).text_size(px(11.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(MUTED_2())).child(title.to_uppercase())
}

fn card() -> gpui::Div {
    div().p(px(14.)).rounded(px(10.)).bg(rgb(BG())).border_1().border_color(rgb(BORDER_SOFT())).flex().flex_col().gap(px(8.))
}

fn note(text: impl Into<SharedString>) -> gpui::Div {
    div().text_size(px(11.5)).line_height(px(16.)).text_color(rgb(MUTED_2())).child(text.into())
}

const AUTHOR_GITHUB: &str = env!("CARGO_PKG_HOMEPAGE");
const AUTHOR_X: &str = "https://x.com/michyaraque";

fn open_dir(dir: &Path, cx: &mut gpui::App) {
    let _ = std::fs::create_dir_all(dir);
    cx.open_with_system(dir);
}

impl Studio {
    pub(super) fn update_settings(&mut self, f: impl FnOnce(&mut Settings), cx: &mut Context<Self>) {
        f(&mut self.settings);
        if let Err(e) = self.settings.save() {
            self.say(tr!(format!("No se pudieron guardar los ajustes: {e}"), format!("Could not save settings: {e}")), Level::Error);
        }
        i18n::set(self.settings.lang());
        crate::theme::set_light(self.settings.light());
        self.settings.apply_to_engine();
        if let Some(b) = &self.backend {
            b.configure(folders(&self.settings));
        }
        cx.notify();
    }

    pub(super) fn pick_folder(&mut self, cx: &mut Context<Self>, then: impl FnOnce(&mut Settings, PathBuf) + 'static) {
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
                let _ = this.update(cx, |s, cx| s.update_settings(|st| then(st, p), cx));
            }
        })
        .detach();
    }

    /// A folder setting: its name, the folder in use as the description, and open / change / back-to-default.
    fn folder_row(
        &mut self,
        id: &'static str,
        title: &str,
        shown: &Path,
        custom: bool,
        set: fn(&mut Settings, Option<PathBuf>),
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let dir = shown.to_path_buf();
        let desc = if custom { display_path(shown) } else { format!("{} · {}", display_path(shown), tr!("predeterminada", "default")) };
        setting_row(
            title,
            desc,
            div()
                .flex()
                .gap(px(6.))
                .child(
                    icon_btn((id, 0usize), "folder-open")
                        .w(px(30.))
                        .h(px(30.))
                        .tooltip(tooltip(tr!("Abrir carpeta", "Open folder")))
                        .on_click(cx.listener(move |_, _, _, cx| open_dir(&dir, cx))),
                )
                .when(custom, |d| {
                    d.child(
                        icon_btn((id, 1usize), "rotate-ccw")
                            .w(px(30.))
                            .h(px(30.))
                            .tooltip(tooltip(tr!("Volver a la predeterminada", "Back to the default")))
                            .on_click(cx.listener(move |this, _, _, cx| this.update_settings(|s| set(s, None), cx))),
                    )
                })
                .child(
                    button((id, 2usize), tr!("Cambiar…", "Change…"))
                        .on_click(cx.listener(move |this, _, _, cx| this.pick_folder(cx, move |s, p| set(s, Some(p))))),
                ),
        )
        .into_any_element()
    }

    fn general_tab(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let chosen = self.settings.language.clone();
        let langs = Lang::ALL
            .into_iter()
            .enumerate()
            .map(|(i, l)| {
                seg(("lang", i), l.native(), chosen.as_deref() == Some(l.code()))
                    .on_click(cx.listener(move |this, _, _, cx| this.update_settings(|s| s.language = Some(l.code().into()), cx)))
            })
            .chain(std::iter::once(
                seg("lang-sys", tr!("Sistema", "System"), chosen.is_none())
                    .on_click(cx.listener(|this, _, _, cx| this.update_settings(|s| s.language = None, cx))),
            ));
        let look = self.settings.appearance.clone();
        let looks: [(Option<&str>, &str); 3] =
            [(Some("dark"), tr!("Oscuro", "Dark")), (Some("light"), tr!("Claro", "Light")), (None, tr!("Sistema", "System"))];
        let appearance = segmented(looks.into_iter().enumerate().map(|(i, (v, name))| {
            seg(("look", i), name, look.as_deref() == v)
                .on_click(cx.listener(move |this, _, _, cx| this.update_settings(|s| s.appearance = v.map(String::from), cx)))
        }));
        let out = self.settings.output_dir.clone().unwrap_or_else(Settings::fallback_output_dir);
        let custom = self.settings.output_dir.is_some();
        div()
            .flex()
            .flex_col()
            .child(setting_row(tr!("Idioma", "Language"), "", segmented(langs)))
            .child(setting_row(tr!("Apariencia", "Appearance"), "", appearance))
            .child(self.folder_row("out", tr!("Carpeta de resultados", "Output folder"), &out, custom, |s, p| s.output_dir = p, cx))
            .into_any_element()
    }

    pub(super) fn start_download(&mut self, label: &str, items: Vec<Item>, cx: &mut Context<Self>) {
        let typed = self.hf_token.read(cx).text().trim().to_string();
        if Some(&typed) != self.settings.hf_token.as_ref() {
            self.update_settings(|s| s.hf_token = (!typed.is_empty()).then_some(typed), cx);
        }
        self.downloads.start(label.to_string(), items, self.settings.hf_token());
        cx.notify();
    }

    pub(super) fn download_rows(&mut self, family: &str, rows: Vec<FileRow>, cx: &mut Context<Self>) {
        let items = rows
            .into_iter()
            .filter_map(|r| {
                let (src, dest) = (r.remote?, r.dest?);
                let name = dest.file_name()?.to_string_lossy().to_string();
                Some(Item { name, url: src.url, dest, repo: src.repo })
            })
            .collect();
        self.start_download(family, items, cx);
    }

    /// The running download of `label`, as a bar with the file, the size and a cancel button.
    pub(super) fn download_progress(&self, label: &str, cx: &mut Context<Self>) -> Option<AnyElement> {
        let p = self.downloads.progress();
        if !p.running || p.family != label {
            return None;
        }
        let frac = (p.total > 0).then(|| p.done as f32 / p.total as f32);
        Some(
            div()
                .flex()
                .items_center()
                .gap(px(10.))
                .pt(px(4.))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap(px(5.))
                        .child(
                            div()
                                .flex()
                                .gap(px(8.))
                                .text_size(px(11.5))
                                .child(div().min_w_0().overflow_hidden().text_ellipsis().whitespace_nowrap().text_color(rgb(TEXT())).child(format!(
                                    "{}/{} · {}",
                                    p.index + 1,
                                    p.count,
                                    p.file
                                )))
                                .child(div().flex_none().font_family(MONO).text_size(px(10.5)).text_color(rgb(MUTED())).child(format!(
                                    "{} / {}",
                                    gb(p.done),
                                    gb(p.total)
                                ))),
                        )
                        .child(super::model_picker::progress_bar(frac, 360.)),
                )
                .child(button(SharedString::from(format!("dl-cancel-{label}")), tr!("Cancelar", "Cancel")).on_click(cx.listener(|this, _, _, cx| {
                    this.downloads.cancel();
                    cx.notify();
                })))
                .into_any_element(),
        )
    }

    /// Any file from Hugging Face ("owner/repo/path", or a huggingface.co link) into the models folder, keeping its
    /// path, or into the LoRA library under `downloads/<repo>`.
    fn download_link(&mut self, as_lora: bool, cx: &mut Context<Self>) {
        let text = self.hf_url.read(cx).text();
        let Some((repo, rev, path)) = parse_hf_link(&text) else {
            self.say(
                tr!("Escribe owner/repo/ruta/al/archivo o pega un enlace de huggingface.co.", "Type owner/repo/path/to/file or paste a huggingface.co link."),
                Level::Error,
            );
            cx.notify();
            return;
        };
        let name = path.rsplit('/').next().unwrap_or(&path).to_string();
        let dest = if as_lora {
            let Some(library) = self.settings.lora_library() else { return };
            library.join("downloads").join(repo.rsplit('/').next().unwrap_or(&repo)).join(&name)
        } else {
            let Some(root) = dotsmith_engine::locations::download_root() else { return };
            root.join(&path)
        };
        let url = format!("https://huggingface.co/{repo}/resolve/{rev}/{path}");
        self.start_download(LINK, vec![Item { name, url, dest, repo }], cx);
    }

    fn models_tab(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let families: Vec<String> = self.model.families.iter().map(|f| f.id.clone()).collect();
        let (ready, total) = (families.iter().filter(|id| self.model.family_ready(id)).count(), families.len());
        let all_ready = total > 0 && ready == total;
        let loras: usize = self.model.runs.iter().map(|r| r.ckpts.len()).sum();
        let models_dir = dotsmith_engine::locations::download_root().unwrap_or_default();
        let library = self.settings.lora_library().unwrap_or_default();
        let hf = self.settings.hf_dirs.clone();
        let busy = self.downloads.progress().running;
        let link_progress = self.download_progress(LINK, cx);
        let catalog_error = self.model.catalog_error.clone();
        let status = div()
            .flex()
            .items_center()
            .gap(px(12.))
            .p(px(14.))
            .rounded(px(10.))
            .bg(rgb(BG()))
            .border_1()
            .border_color(if all_ready { rgb(BORDER_SOFT()) } else { rgba(0xe8c06a55) })
            .child(icon(if all_ready { "circle-check-big" } else { "triangle-alert" }).size(px(18.)).text_color(rgb(if all_ready { BLUE() } else { WARN() })))
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(
                        div()
                            .text_size(px(13.))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(rgb(TEXT()))
                            .child(tr!(format!("{ready} de {total} modelos base listos"), format!("{ready} of {total} base models ready"))),
                    )
                    .child(note(tr!(format!("{loras} LoRAs en tu biblioteca"), format!("{loras} LoRAs in your library")))),
            )
            .child(
                div()
                    .kb("open-hub")
                    .h(px(32.))
                    .px(px(14.))
                    .flex()
                    .items_center()
                    .gap(px(7.))
                    .rounded(px(8.))
                    .bg(rgb(ACCENT()))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(ACCENT_HOVER())))
                    .child(icon("box").size(px(14.)).text_color(rgb(ACCENT_INK())))
                    .child(
                        div()
                            .text_size(px(12.5))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(rgb(ACCENT_INK()))
                            .child(tr!("Gestionar modelos", "Manage models")),
                    )
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.settings_open = false;
                        this.open_hub(None, cx);
                    })),
            );
        div()
            .flex()
            .flex_col()
            .children(catalog_error.map(|e| {
                card()
                    .mb(px(14.))
                    .border_color(rgba(0xe07a6a66))
                    .child(div().text_size(px(12.)).text_color(rgb(TEXT())).child({
                        let file = crate::settings::config_dir().map(|d| display_path(&d.join("models.json"))).unwrap_or_default();
                        tr!(format!("Se ignoró {file}:"), format!("{file} was ignored:"))
                    }))
                    .child(div().font_family(MONO).text_size(px(10.5)).text_color(rgb(DANGER())).child(e))
            }))
            .child(status)
            .child(group(tr!("Carpetas", "Folders")))
            .child(self.folder_row("models", tr!("Modelos base", "Base models"), &models_dir, self.settings.models_dir.is_some(), |s, p| s.models_dir = p, cx))
            .child(self.folder_row("loras", "LoRAs", &library, self.settings.lora_dir.is_some(), |s, p| s.lora_dir = p, cx))
            .child(setting_row(
                tr!("Otras carpetas de modelos", "Other model folders"),
                tr!("Carpetas descargadas de Hugging Face, tal cual.", "Folders downloaded from Hugging Face, as they are."),
                button_i("hf-add", "plus", tr!("Añadir…", "Add…")).on_click(cx.listener(|this, _, _, cx| {
                    this.pick_folder(cx, |s, p| {
                        if !s.hf_dirs.contains(&p) {
                            s.hf_dirs.push(p)
                        }
                    })
                })),
            ))
            .children(hf.into_iter().enumerate().map(|(i, d)| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .pl(px(12.))
                    .pb(px(8.))
                    .child(icon("folder-open").size(px(13.)).text_color(rgb(MUTED_2())))
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
                            .child(display_path(&d)),
                    )
                    .child(icon_btn(("hf-del", i), "x").tooltip(tooltip(tr!("Quitar", "Remove"))).on_click(cx.listener(move |this, _, _, cx| {
                        this.update_settings(
                            |s| {
                                if i < s.hf_dirs.len() {
                                    s.hf_dirs.remove(i);
                                }
                            },
                            cx,
                        )
                    })))
            }))
            .child(group("Hugging Face"))
            .child(setting_row(
                tr!("Token de acceso", "Access token"),
                tr!("Solo para modelos con licencia que aceptar.", "Only for models with a license to accept."),
                div().w(px(260.)).child(self.hf_token.clone()),
            ))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .py(px(12.))
                    .border_t_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(3.))
                            .child(div().text_size(px(13.)).text_color(rgb(TEXT())).child(tr!("Descargar un archivo", "Download a file")))
                            .child(note(tr!("owner/repo/ruta o un enlace de huggingface.co", "owner/repo/path or a huggingface.co link"))),
                    )
                    .child(self.hf_url.clone())
                    .child(div().flex().gap(px(6.)).when(!busy, |d| {
                        d.child(
                            button_i("dl-link-lora", "layers", tr!("Como LoRA", "As a LoRA"))
                                .on_click(cx.listener(|this, _, _, cx| this.download_link(true, cx))),
                        )
                        .child(
                            button_i("dl-link-models", "download", tr!("A modelos base", "To base models"))
                                .on_click(cx.listener(|this, _, _, cx| this.download_link(false, cx))),
                        )
                    }))
                    .children(link_progress),
            )
            .into_any_element()
    }

    fn gpu_tab(&mut self, cx: &mut Context<Self>) -> AnyElement {
        use dotsmith_engine::core::memory::{HostMemory, Placement};
        let gpu = self.model.gpu.clone();
        let info: AnyElement = match &gpu {
            Some(g) if g.ok => div()
                .flex()
                .items_center()
                .gap(px(12.))
                .p(px(14.))
                .rounded(px(10.))
                .bg(rgb(BG()))
                .border_1()
                .border_color(rgb(BORDER_SOFT()))
                .child(
                    div()
                        .w(px(36.))
                        .h(px(36.))
                        .rounded(px(9.))
                        .bg(rgb(RAISED()))
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(icon("cpu").size(px(18.)).text_color(rgb(ACCENT()))),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(div().text_size(px(13.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(g.name.clone()))
                        .child(note(format!("{:.0} GB VRAM", g.vram_mb as f64 / 1024.))),
                )
                .into_any_element(),
            Some(g) => card()
                .border_color(rgba(0xe07a6a66))
                .child(
                    div()
                        .flex()
                        .items_start()
                        .gap(px(10.))
                        .child(icon("triangle-alert").size(px(18.)).text_color(rgb(DANGER())))
                        .child(div().flex_1().min_w_0().text_size(px(12.)).line_height(px(17.)).text_color(rgb(TEXT())).child(g.error.clone())),
                )
                .into_any_element(),
            None => note(tr!("Comprobando la GPU…", "Checking the GPU…")).into_any_element(),
        };
        let low = self.settings.low_vram;
        let effective = gpu.as_ref().is_some_and(|g| g.low_vram);
        let m = self.settings.memory.clone();
        let low_opts: [(Option<bool>, &str); 3] = [(None, tr!("Auto", "Auto")), (Some(true), tr!("Sí", "On")), (Some(false), "Off")];
        let mut tags: Vec<String> = self.model.files.iter().flat_map(|(_, rows)| rows.iter().filter_map(|r| r.precision.clone())).collect();
        tags.sort();
        tags.dedup();
        let precisions: Vec<(Option<String>, String)> =
            std::iter::once((None, tr!("8 bits", "8-bit").to_string())).chain(tags.into_iter().map(|t| (Some(t.clone()), t))).collect();
        let fam_has = |tag: &str| self.model.files_of(&self.model.family).iter().any(|r| r.precision.as_deref() == Some(tag) && r.found);
        let precision_desc = match &m.precision {
            Some(t) if !fam_has(t) => tr!(
                format!("El modelo actual no tiene archivo {t}: se usa el de 8 bits."),
                format!("The current model has no {t} file: the 8-bit one is used.")
            ),
            Some(_) => tr!("Pesos originales: el doble de memoria, algo más lento.", "Original weights: twice the memory, a bit slower.").to_string(),
            None => tr!("La mitad de memoria y más rápido; diferencias mínimas.", "Half the memory and faster; tiny differences.").to_string(),
        };
        let placements: [(Placement, &str); 3] = [(Placement::Auto, "Auto"), (Placement::Gpu, "GPU"), (Placement::Stream, "RAM")];
        let hosts: [(HostMemory, &str); 3] =
            [(HostMemory::Auto, "Auto"), (HostMemory::Pinned, tr!("Rápida", "Fast")), (HostMemory::Staged, tr!("Ahorro", "Saver"))];
        let limits: Vec<Option<u64>> = vec![None, Some(4096), Some(6144), Some(8192), Some(10240), Some(12288), Some(16384), Some(20480)];
        let limit_ix = limits.iter().position(|l| *l == m.vram_limit_mb);
        let limit_text = match m.vram_limit_mb {
            None => tr!("Sin límite", "No limit").to_string(),
            Some(l) => format!("{:.0} GB", l as f64 / 1024.),
        };
        let step_limit = move |d: i32| {
            let limits = limits.clone();
            move |this: &mut Studio, _: &gpui::ClickEvent, _: &mut gpui::Window, cx: &mut Context<Studio>| {
                let i = (limit_ix.unwrap_or(0) as i32 + d).clamp(0, limits.len() as i32 - 1) as usize;
                let v = limits[i];
                this.update_settings(|s| s.memory.vram_limit_mb = v, cx);
            }
        };
        let placed: Vec<AnyElement> = gpu
            .as_ref()
            .map(|g| g.memory.clone())
            .unwrap_or_default()
            .into_iter()
            .map(|p| {
                let what = match p.component.as_str() {
                    "dit" => tr!("Modelo de difusión", "Diffusion model"),
                    "text_encoder" => tr!("Codificador de texto", "Text encoder"),
                    other => other,
                }
                .to_string();
                let what = if p.format.is_empty() { what } else { format!("{what} ({})", p.format) };
                let text = if p.streamed == 0 {
                    tr!(format!("{what}: las {} capas en la GPU", p.layers), format!("{what}: all {} layers on the GPU", p.layers))
                } else {
                    tr!(
                        format!("{what}: {} de {} capas desde la RAM ({:.1} GB)", p.streamed, p.layers, p.locked_mb as f64 / 1024.),
                        format!("{what}: {} of {} layers from RAM ({:.1} GB)", p.streamed, p.layers, p.locked_mb as f64 / 1024.)
                    )
                };
                div().text_size(px(12.)).text_color(rgb(TEXT())).child(text).into_any_element()
            })
            .collect();
        let loaded = !placed.is_empty();
        div()
            .flex()
            .flex_col()
            .child(info)
            .child(group(tr!("Memoria", "Memory")))
            .child(setting_row(
                tr!("Modo de poca VRAM", "Low-VRAM mode"),
                tr!(
                    format!("Más lento, usa menos memoria. Ahora: {}.", if effective { "sí" } else { "no" }),
                    format!("Slower, uses less memory. Now: {}.", if effective { "on" } else { "off" })
                ),
                segmented(low_opts.into_iter().enumerate().map(|(i, (v, name))| {
                    seg(("lowvram", i), name, low == v).on_click(cx.listener(move |this, _, _, cx| this.update_settings(|s| s.low_vram = v, cx)))
                })),
            ))
            .when(precisions.len() > 1, |d| {
                d.child(setting_row(
                    tr!("Precisión de los pesos", "Weight precision"),
                    precision_desc,
                    segmented(precisions.into_iter().enumerate().map(|(i, (v, name))| {
                        let on = m.precision == v;
                        seg(("mem-prec", i), &name, on).on_click(cx.listener(move |this, _, _, cx| {
                            let v = v.clone();
                            this.update_settings(|s| s.memory.precision = v, cx)
                        }))
                    })),
                ))
            })
            .child(setting_row(
                tr!("Dónde van los pesos", "Where weights live"),
                tr!("Desde la RAM: misma calidad con menos VRAM.", "From RAM: same quality with less VRAM."),
                segmented(placements.into_iter().enumerate().map(|(i, (v, name))| {
                    seg(("mem-place", i), name, m.placement == v)
                        .on_click(cx.listener(move |this, _, _, cx| this.update_settings(|s| s.memory.placement = v, cx)))
                })),
            ))
            .child(setting_row(
                tr!("Límite de VRAM", "VRAM limit"),
                tr!("Lo máximo que Dotsmith puede ocupar.", "The most Dotsmith may take."),
                div()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(icon_btn(("vram-lim", 0usize), "minus").on_click(cx.listener(step_limit(-1))))
                    .child(div().w(px(76.)).flex().justify_center().font_family(MONO).text_size(px(12.)).text_color(rgb(TEXT())).child(limit_text))
                    .child(icon_btn(("vram-lim", 1usize), "plus").on_click(cx.listener(step_limit(1)))),
            ))
            .child(setting_row(
                tr!("RAM para el streaming", "Streaming RAM"),
                tr!("Rápida reserva RAM mientras el modelo está cargado.", "Fast locks RAM while the model is loaded."),
                segmented(hosts.into_iter().enumerate().map(|(i, (v, name))| {
                    seg(("mem-host", i), name, m.host == v).on_click(cx.listener(move |this, _, _, cx| this.update_settings(|s| s.memory.host = v, cx)))
                })),
            ))
            .child(setting_row(
                tr!("Imágenes por pasada", "Images per pass"),
                tr!("Más imágenes comparten cada envío de pesos.", "More images share every weight upload."),
                segmented([1usize, 2, 4, 8].into_iter().map(|n| {
                    seg(("mem-batch", n), &n.to_string(), m.batch_max == n)
                        .on_click(cx.listener(move |this, _, _, cx| this.update_settings(|s| s.memory.batch_max = n, cx)))
                })),
            ))
            .child(
                div().pt(px(16.)).child(
                    card()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .child(
                                    div()
                                        .flex_1()
                                        .text_size(px(12.))
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(rgb(MUTED()))
                                        .child(tr!("Última carga", "Last load")),
                                )
                                .when(loaded, |d| {
                                    d.child(button_i("mem-reload", "rotate-ccw", tr!("Recargar ahora", "Reload now")).on_click(cx.listener(
                                        |this, _, _, cx| {
                                            this.send(json!({"cmd": "unload"}));
                                            this.send(json!({"cmd": "load"}));
                                            cx.notify();
                                        },
                                    )))
                                }),
                        )
                        .when(loaded, |d| d.children(placed))
                        .child(note(if loaded {
                            tr!("Los cambios se aplican en la próxima carga.", "Changes apply on the next load.")
                        } else {
                            tr!("Ningún modelo cargado.", "No model loaded.")
                        })),
                ),
            )
            .into_any_element()
    }

    fn shortcuts_tab(&mut self) -> AnyElement {
        let rows: [(&str, &str); 14] = [
            (tr!("Buscar cualquier cosa", "Search anything"), "Ctrl + K"),
            (tr!("Control siguiente / anterior", "Next / previous control"), tr!("Tab / Mayús + Tab", "Tab / Shift + Tab")),
            (tr!("Activar el control", "Activate the control"), tr!("Intro o Espacio", "Enter or Space")),
            (tr!("Deshacer el último borrado", "Undo the last delete"), "Ctrl + Z"),
            (tr!("Generar", "Generate"), "Ctrl + Enter"),
            (tr!("Elegir modelo", "Choose a model"), "Ctrl + L"),
            (tr!("Detener o cerrar una ventana", "Stop or close a window"), "Esc"),
            (tr!("Pegar una imagen como origen", "Paste an image as the source"), "Ctrl + V"),
            (tr!("Copiar el resultado", "Copy the result"), "Ctrl + C"),
            (tr!("Copiar el resultado ampliado", "Copy the result enlarged"), tr!("Mayús + clic en Copiar", "Shift + click Copy")),
            (tr!("Resultado anterior / siguiente", "Previous / next result"), "← / →"),
            (tr!("Mover el resultado a la papelera", "Move the result to the Recycle Bin"), tr!("Supr", "Delete")),
            (tr!("Restablecer el zoom", "Reset the zoom"), tr!("Doble clic", "Double-click")),
            (tr!("Ajustes", "Settings"), "Ctrl + ,"),
        ];
        div().flex().flex_col().children(rows.into_iter().map(|(what, keys)| setting_row(what, "", kbd(keys)))).into_any_element()
    }

    fn about_tab(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let logs = crate::settings::log_dir();
        let config = crate::settings::config_dir();
        div().flex().flex_col()
            .child(div().flex().items_center().gap(px(14.)).pb(px(18.))
                .child(gpui::img(self.logo.clone()).w(px(52.)).h(px(52.)))
                .child(div().flex().flex_col().gap(px(3.))
                    .child(div().text_size(px(20.)).font_weight(FontWeight::BOLD).text_color(rgb(TEXT())).child("Dotsmith"))
                    .child(note(format!("{} {} · {}", tr!("Versión", "Version"), env!("CARGO_PKG_VERSION"),
                        tr!("todo se ejecuta en tu PC", "everything runs on your PC"))))))
            .child(setting_row(tr!("Creado por", "Made by"), env!("CARGO_PKG_AUTHORS"),
                div().flex().gap(px(8.))
                    .child(button_i("author-github", "external-link", "GitHub").on_click(|_, _, cx| cx.open_url(AUTHOR_GITHUB)))
                    .child(button_i("author-x", "external-link", "X").on_click(|_, _, cx| cx.open_url(AUTHOR_X)))))
            .child(setting_row(tr!("Registros", "Logs"), tr!("Para adjuntar a un informe de error.", "To attach to a bug report."),
                button_i("open-logs", "file-text", tr!("Abrir", "Open")).on_click(cx.listener(move |_, _, _, cx| open_dir(&logs, cx)))))
            .when_some(config, |d, c| d.child(setting_row(tr!("Carpeta de ajustes", "Settings folder"), "settings.json · models.json",
                button_i("open-config", "folder-open", tr!("Abrir", "Open")).on_click(cx.listener(move |_, _, _, cx| open_dir(&c, cx))))))
            .child(group(tr!("Licencias", "Licenses")))
            .child(note(tr!(
                "Dotsmith: MIT. Iconos: Lucide (ISC). Interfaz: GPUI (Apache 2.0). Motor: candle (MIT / Apache 2.0). Los modelos base y los LoRAs tienen sus propias licencias: revísalas antes de un uso comercial.",
                "Dotsmith: MIT. Icons: Lucide (ISC). Interface: GPUI (Apache 2.0). Engine: candle (MIT / Apache 2.0). Base models and LoRAs carry their own licenses: check them before commercial use.")))
            .into_any_element()
    }

    pub(super) fn settings_view(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let tab = self.settings_tab;
        let body = match tab {
            Tab::General => self.general_tab(cx),
            Tab::Models => self.models_tab(cx),
            Tab::Gpu => self.gpu_tab(cx),
            Tab::Shortcuts => self.shortcuts_tab(),
            Tab::About => self.about_tab(cx),
        };
        let attention = |t: Tab, s: &Studio| match t {
            Tab::Models => !s.model.files.is_empty() && !s.model.family_ready(&s.model.family),
            Tab::Gpu => s.model.gpu.as_ref().is_some_and(|g| !g.ok),
            _ => false,
        };
        let nav: Vec<AnyElement> = Tab::ALL
            .into_iter()
            .map(|t| {
                let on = t == tab;
                let warn = attention(t, self);
                div()
                    .kb(t.title())
                    .h(px(32.))
                    .px(px(10.))
                    .flex()
                    .items_center()
                    .gap(px(9.))
                    .rounded(px(7.))
                    .cursor_pointer()
                    .when(on, |d| d.bg(rgb(RAISED())))
                    .when(!on, |d| d.hover(|s| s.bg(rgb(PANEL()))))
                    .child(icon(t.glyph()).size(px(15.)).text_color(rgb(if on { ACCENT() } else { MUTED() })))
                    .child(div().flex_1().text_size(px(12.5)).text_color(rgb(if on { TEXT() } else { MUTED() })).child(t.title()))
                    .when(warn, |d| d.child(div().w(px(7.)).h(px(7.)).rounded_full().bg(rgb(WARN()))))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.settings_tab = t;
                        cx.notify();
                    }))
                    .into_any_element()
            })
            .collect();
        let card = div()
            .id("settings")
            .occlude()
            .w(px(900.))
            .h(px(640.))
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
                    .w(px(210.))
                    .flex_none()
                    .p(px(10.))
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .bg(rgb(BG()))
                    .border_r_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .child(
                        div()
                            .px(px(10.))
                            .pt(px(8.))
                            .pb(px(12.))
                            .text_size(px(15.))
                            .font_weight(FontWeight::BOLD)
                            .text_color(rgb(TEXT()))
                            .child(tr!("Ajustes", "Settings")),
                    )
                    .children(nav),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .h(px(56.))
                            .flex_none()
                            .px(px(28.))
                            .flex()
                            .items_center()
                            .gap(px(12.))
                            .child(div().flex_1().text_size(px(16.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(tab.title()))
                            .child(
                                div()
                                    .kb("settings-close")
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
                                        this.settings_open = false;
                                        cx.notify();
                                    })),
                            ),
                    )
                    .child(div().id("settings-body").flex_1().min_h_0().overflow_y_scroll().px(px(28.)).pb(px(24.)).child(body)),
            );
        self.modal_layer("settings-veil", card, |s, _, _| s.settings_open = false, cx)
    }
}

#[cfg(test)]
mod tests {
    use super::parse_hf_link;

    #[test]
    fn reads_repo_paths_and_links() {
        let want = Some(("owner/repo".to_string(), "main".to_string(), "loras/x.safetensors".to_string()));
        assert_eq!(parse_hf_link("owner/repo/loras/x.safetensors"), want);
        assert_eq!(parse_hf_link("https://huggingface.co/owner/repo/resolve/main/loras/x.safetensors"), want);
        assert_eq!(parse_hf_link("huggingface.co/owner/repo/blob/main/loras/x.safetensors"), want);
        assert_eq!(parse_hf_link("owner/repo"), None);
    }
}

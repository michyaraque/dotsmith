//! The "Personaje" mode UI: its options (camera, four directions or one, cell size) and the animated viewer that plays
//! every direction of a saved walk sheet. Generation itself is jobs::creature_mode; frames come from crate::anim.
use gpui::{AnyElement, Context, FontWeight, canvas, div, prelude::*, px, rgb, rgba};
use image::RgbaImage;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{Level, Studio, button, button_i, chip, icon_btn, label, paint_checker, paint_pixels, reveal_in_explorer, toggle};
use crate::anim;
use crate::jobs::{CREATURE_CAMERAS, DIRS};
use crate::theme::*;
use crate::tr;

fn camera_label(i: usize) -> &'static str {
    if i == 0 { tr!("Oblicua clásica", "Classic oblique") } else { tr!("Cenital 3/4", "Top-down 3/4") }
}
/// Frame sizes (w, h) per camera, the most common cells of the training sheets (krea2_creatures_x8_v1): Tibia oblique
/// sheets have square cells (32-44 px), the top-down 3/4 ones (Drakantos) tall cells, 32 x 56 the most frequent.
/// Never past 44 px wide: at x8 that is the widest sheet the LoRA saw (1408 px).
// square cells only: the 8-frame walk sheets (walks8_x8_v2) were trained at 4x2 cells of 32, 48 or 64 px (x8 ->
// 1024x512, 1536x768, 2048x1024); tall cells (32x56) are outside that data
pub const CELLS: [[(u32, u32); 3]; 2] = [[(32, 32), (48, 48), (64, 64)], [(32, 32), (48, 48), (64, 64)]];
fn dir_label_at(i: usize) -> &'static str {
    match i {
        0 => tr!("Sur", "South"),
        1 => tr!("Oeste", "West"),
        2 => tr!("Norte", "North"),
        _ => tr!("Este", "East"),
    }
}

fn dir_label(d: &str) -> &'static str {
    DIRS.iter().position(|x| *x == d).map(dir_label_at).unwrap_or("")
}

/// A saved creature sheet as the viewer needs it.
#[derive(Clone)]
pub struct SheetInfo {
    pub cell: u32,
    pub cell_h: u32,
    /// drawn frames per direction (sheet columns)
    pub frames: usize,
    pub dirs: Vec<String>,
    /// columns in playback order (LoRA walk formats, e.g. [1, 0, 2, 0]); the built-in format plays 0..frames
    pub play: Vec<usize>,
}

impl SheetInfo {
    pub fn of(json: &Value) -> Option<SheetInfo> {
        (json["mode"].as_str() == Some("creature")).then(|| {
            let frames = json["frames"].as_u64().unwrap_or(8) as usize;
            let play: Vec<usize> = json["play"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).filter(|&v| v < frames).collect())
                .filter(|p: &Vec<usize>| !p.is_empty())
                .unwrap_or_else(|| (0..frames).collect());
            SheetInfo {
                cell: json["cell"].as_u64().unwrap_or(32) as u32,
                cell_h: json["cell_h"].as_u64().or(json["cell"].as_u64()).unwrap_or(32) as u32,
                frames,
                dirs: json["dirs"].as_array().map(|a| a.iter().filter_map(|d| d.as_str().map(String::from)).collect()).unwrap_or_default(),
                play,
            }
        })
    }

    /// Animation length (frames played per loop).
    pub fn len(&self) -> usize {
        self.play.len().max(1)
    }

    /// Sheet column of animation frame `f`.
    pub fn col(&self, f: usize) -> u32 {
        self.play.get(f % self.len()).copied().unwrap_or(0) as u32
    }
}

/// GIFs of a saved sheet at `fps`, pixels x4: one with every direction side by side and one per direction, next to
/// the sheet (<stem>_4dir_<fps>fps.gif, <stem>_<dir>_<fps>fps.gif). Returns the side-by-side one.
pub fn export_gifs(png: &Path, sheet: &RgbaImage, info: &SheetInfo, fps: u32) -> anyhow::Result<PathBuf> {
    let stem = png.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let (c, ch) = (info.cell, info.cell_h);
    let rows: Vec<Vec<RgbaImage>> = (0..info.dirs.len() as u32).map(|r| (0..info.len()).map(|f| anim::frame(sheet, c, ch, r, info.col(f))).collect()).collect();
    for (d, frames) in info.dirs.iter().zip(&rows) {
        std::fs::write(png.with_file_name(format!("{stem}_{d}_{fps}fps.gif")), anim::gif(frames, fps, 4)?)?;
    }
    let side: Vec<RgbaImage> = (0..info.len())
        .map(|f| {
            let mut im = RgbaImage::new(c * rows.len() as u32, ch);
            for (i, r) in rows.iter().enumerate() {
                image::imageops::overlay(&mut im, &r[f], (i as u32 * c) as i64, 0);
            }
            im
        })
        .collect();
    let out = png.with_file_name(format!("{stem}_{}dir_{fps}fps.gif", rows.len()));
    std::fs::write(&out, anim::gif(&side, fps, 4)?)?;
    Ok(out)
}

impl Studio {
    /// Mode options under the description.
    pub(super) fn creature_options(&mut self, cx: &mut Context<Self>) -> AnyElement {
        if let Some(w) = self.model.walk.clone() {
            return self.lora_walk_options(&w, cx);
        }
        let f = self.form();
        let (camera, all, dir, cell) = (f.camera.min(1), f.all_dirs, f.dir, f.size.min(2));
        let supported = self.model.families.iter().any(|f| f.id == self.model.family && f.capabilities.iter().any(|c| c == "walk_sheet"));
        div()
            .flex()
            .flex_col()
            .when(!supported, |d| {
                d.child(
                    div()
                        .mt(px(14.))
                        .p(px(10.))
                        .flex()
                        .items_center()
                        .gap(px(10.))
                        .rounded(px(8.))
                        .bg(rgba(0xe8c06a1a))
                        .border_1()
                        .border_color(rgba(0xe8c06a55))
                        .child(
                            div()
                                .flex_1()
                                .text_size(px(11.5))
                                .line_height(px(16.))
                                .text_color(rgb(WARN()))
                                .child(tr!("Este modelo no hace hojas de caminar.", "This model can't make walk sheets.")),
                        )
                        .child(button("walk-pick", tr!("Cambiar modelo", "Change model")).on_click(cx.listener(|this, _, _, cx| {
                            this.open_model_picker();
                            cx.notify();
                        }))),
                )
            })
            .child(div().pt(px(14.)).pb(px(6.)).child(label(tr!("Cámara", "Camera"))))
            .child(div().flex().gap(px(6.)).children((0..2).map(|i| {
                chip(("ccam", i), camera_label(i), camera == i).on_click(cx.listener(move |this, _, _, cx| {
                    this.form_mut().camera = i;
                    cx.notify();
                }))
            })))
            .child(div().pt(px(10.)).child(toggle("alldirs", tr!("Las 4 direcciones", "All 4 directions"), "", all).on_click(cx.listener(|this, _, _, cx| {
                let f = this.form_mut();
                f.all_dirs = !f.all_dirs;
                cx.notify();
            }))))
            .when(!all, |d| {
                d.child(div().flex().gap(px(6.)).pt(px(4.)).children((0..4).map(|i| {
                    chip(("cdir", i), dir_label_at(i), dir == i).on_click(cx.listener(move |this, _, _, cx| {
                        this.form_mut().dir = i;
                        cx.notify();
                    }))
                })))
            })
            .child(div().pt(px(14.)).pb(px(6.)).child(label(tr!("Tamaño del frame", "Frame size"))))
            .child(div().flex().gap(px(6.)).children(CELLS[camera].iter().enumerate().map(|(i, &(w, h))| {
                chip(("ccell", i), &if w == h { format!("{w} px") } else { format!("{w} × {h}") }, cell == i).on_click(cx.listener(move |this, _, _, cx| {
                    this.form_mut().size = i;
                    cx.notify();
                }))
            })))
            .into_any_element()
    }

    /// Options of a LoRA that declares its own walk format: its kinds, and what the sheet will hold (fixed by the LoRA).
    fn lora_walk_options(&mut self, w: &dotsmith_engine::catalog::WalkFormat, cx: &mut Context<Self>) -> AnyElement {
        let lang = if crate::i18n::en() { "en" } else { "es" };
        let kind = self.form().kind.min(w.kinds.len().saturating_sub(1));
        let (cw, ch) = w.cell(self.model.scale);
        let derived = w.derive.len();
        let dir_names = |ds: &mut dyn Iterator<Item = &String>| ds.map(|d| dir_label(d)).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(", ");
        let drawn = dir_names(&mut w.row_dirs.iter());
        let made = dir_names(&mut w.derive.keys());
        div()
            .flex()
            .flex_col()
            .when(!w.kinds.is_empty(), |d| {
                d.child(div().pt(px(14.)).pb(px(6.)).child(label(tr!("Tipo", "Kind")))).child(div().flex().flex_wrap().gap(px(6.)).children(
                    w.kinds.iter().enumerate().map(|(i, k)| {
                        chip(("wkind", i), &k.label.get(lang), kind == i).on_click(cx.listener(move |this, _, _, cx| {
                            this.form_mut().kind = i;
                            cx.notify();
                        }))
                    }),
                ))
            })
            .child(div().pt(px(12.)).text_size(px(11.5)).line_height(px(16.)).text_color(rgb(MUTED_2())).child(tr!(
                format!(
                    "Formato del LoRA: {} frames por dirección de {cw} × {ch} px, en una sola imagen. Dibuja: {drawn}.{}",
                    w.cols,
                    if derived > 0 { format!(" Se derivan: {made}.") } else { String::new() }
                ),
                format!(
                    "LoRA format: {} frames per direction of {cw} × {ch} px, in one image. Drawn: {drawn}.{}",
                    w.cols,
                    if derived > 0 { format!(" Derived: {made}.") } else { String::new() }
                )
            )))
            .into_any_element()
    }

    /// The request fields this mode adds.
    pub(super) fn creature_request(&self, req: &mut Value) {
        let f = self.form();
        req["kind"] = f.kind.into();
        req["camera"] = CREATURE_CAMERAS[f.camera.min(1)].into();
        let (w, h) = CELLS[f.camera.min(1)][f.size.min(2)];
        req["cell"] = w.into();
        req["cell_h"] = h.into();
        req["all_dirs"] = f.all_dirs.into();
        req["dir"] = f.dir.into();
    }

    /// Every direction of a saved sheet walking at once, with play / pause, frame stepping and speed.
    pub(super) fn creature_player(&mut self, png: PathBuf, sheet: Arc<RgbaImage>, info: SheetInfo, scale: f32, cx: &mut Context<Self>) -> AnyElement {
        let a = &self.anim;
        let (frame, playing, fps) = (a.frame % info.len(), a.playing, a.fps);
        let n = info.len();
        let tiles = info
            .dirs
            .iter()
            .enumerate()
            .map(|(row, d)| {
                let im = anim::frame(&sheet, info.cell, info.cell_h, row as u32, info.col(frame));
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .gap(px(6.))
                    .flex_1()
                    .min_w_0()
                    .child(
                        div()
                            .relative()
                            .w_full()
                            .h(px(200.))
                            .rounded(px(8.))
                            .overflow_hidden()
                            .border_1()
                            .border_color(rgb(BORDER_SOFT()))
                            .child(canvas(|_, _, _| (), |b, _, window, _| paint_checker(b, 12., window)).absolute().size_full())
                            .child(canvas(|_, _, _| (), move |b, _, window, _| paint_pixels(b, &im, scale, window)).absolute().size_full()),
                    )
                    .child(div().text_size(px(11.5)).text_color(rgb(MUTED())).child(dir_label(d)))
            })
            .collect::<Vec<_>>();
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .flex()
            .flex_col()
            .p(px(18.))
            .gap(px(14.))
            .bg(rgb(BG()))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(icon_btn("cplay", if playing { "pause" } else { "play" }).on_click(cx.listener(|this, _, _, cx| {
                        this.anim.playing = !this.anim.playing;
                        cx.notify();
                    })))
                    .child(icon_btn(("cstep", 0usize), "minus").on_click(cx.listener(move |this, _, _, cx| {
                        this.anim.step(-1, n);
                        cx.notify();
                    })))
                    .child(icon_btn(("cstep", 1usize), "plus").on_click(cx.listener(move |this, _, _, cx| {
                        this.anim.step(1, n);
                        cx.notify();
                    })))
                    .child(div().font_family(MONO).text_size(px(11.5)).text_color(rgb(TEXT())).child(format!("frame {} / {n}", frame + 1)))
                    .child(div().flex_1())
                    .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child("fps"))
                    .children([4u32, 6, 8, 12].into_iter().map(|v| {
                        chip(("cfps", v as usize), &v.to_string(), fps == v).on_click(cx.listener(move |this, _, _, cx| {
                            this.anim.fps = v;
                            cx.notify();
                        }))
                    }))
                    .child({
                        let (png, sheet2) = (png.clone(), sheet.clone());
                        let info2 = info.clone();
                        button_i("cgif", "image", tr!("Exportar GIF", "Export GIF")).on_click(cx.listener(move |this, _, _, cx| {
                            match export_gifs(&png, &sheet2, &info2, this.anim.fps) {
                                Ok(p) => {
                                    let (fps, name) = (this.anim.fps, p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());
                                    this.say(tr!(format!("GIF exportado a {fps} fps: {name}"), format!("GIF exported at {fps} fps: {name}")), Level::Ok);
                                    reveal_in_explorer(&p);
                                }
                                Err(e) => this.say(tr!(format!("No se pudo exportar el GIF: {e}"), format!("Could not export the GIF: {e}")), Level::Error),
                            }
                            cx.notify();
                        }))
                    }),
            )
            .child(div().flex().gap(px(12.)).children(tiles))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(6.))
                    .child(div().text_size(px(10.5)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(MUTED())).child(tr!(
                        format!("Hoja · {} × {} frames de {} × {} px", info.frames, info.dirs.len(), info.cell, info.cell_h),
                        format!("Sheet · {} × {} frames of {} × {} px", info.frames, info.dirs.len(), info.cell, info.cell_h)
                    )))
                    .child(
                        div()
                            .relative()
                            .w_full()
                            .h(px(150.))
                            .rounded(px(8.))
                            .overflow_hidden()
                            .border_1()
                            .border_color(rgb(BORDER_SOFT()))
                            .child(canvas(|_, _, _| (), |b, _, window, _| paint_checker(b, 12., window)).absolute().size_full())
                            .child(canvas(|_, _, _| (), move |b, _, window, _| paint_pixels(b, &sheet, scale, window)).absolute().size_full()),
                    ),
            )
            .into_any_element()
    }
}

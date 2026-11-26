//! The title bar's Tools menu: a closed 3x3 room assembled from any 2x2 wall set, pack splitting and magenta
//! background removal, on the selected result or on PNGs picked from disk.
use gpui::{AnyElement, Context, PathPromptOptions, div, prelude::*, px, rgb};
use image::RgbaImage;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

use super::{Keyboard, Level, Studio};
use crate::icons::icon;
use crate::pixels;
use crate::theme::*;
use crate::tr;

const CELL: u32 = 64;

/// The four 64 px pieces (pile, north, east, corner = cells (0,0) (1,0) (0,1) (1,1) of every v16 / krea2 wall sheet)
/// of a wall-set image at native size or any exact upscale. The scale comes from the .json next to it when it says,
/// else native if it is at most 3 cells wide, else the 2x2 (square) or 3-column upscale. Flat #FF00FF counts as empty.
fn wall_pieces(png: &Path) -> Result<Vec<RgbaImage>, String> {
    let im = image::open(png).map_err(|e| e.to_string())?.to_rgba8();
    let (w, h) = im.dimensions();
    let meta: Value = std::fs::read_to_string(png.with_extension("json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    // columns from a training caption ("format wall set 3x2") when there is one
    let cols = std::fs::read_to_string(png.with_extension("txt"))
        .ok()
        .and_then(|t| t.split_once("format wall set ").and_then(|(_, r)| r.chars().next()?.to_digit(10)));
    let s = match meta["scale"].as_u64().filter(|_| meta["mode"].is_null()) {
        Some(s) => s as u32,
        None if w <= 3 * CELL => 1,
        None => match cols {
            Some(c) => w / (c * CELL),
            None if w == h => w / (2 * CELL),
            None => w / (3 * CELL),
        },
    }
    .max(1);
    if w < 2 * CELL * s || h < 2 * CELL * s || w % s != 0 || h % s != 0 {
        return Err(tr!(
            format!("{w} × {h} px no es un set de muros 2×2 de 64 px (escala x{s})."),
            format!("{w} × {h} px is not a 2×2 wall set of 64 px cells (scale x{s}).")
        ));
    }
    let pieces: Vec<RgbaImage> = (0..4u32)
        .map(|i| {
            let (cx, cy) = ((i % 2) * CELL, (i / 2) * CELL);
            RgbaImage::from_fn(CELL, CELL, |x, y| {
                let p = *im.get_pixel((cx + x) * s + s / 2, (cy + y) * s + s / 2);
                if p.0[3] < 128 || (p.0[0] > 230 && p.0[1] < 40 && p.0[2] > 230) {
                    image::Rgba([0, 0, 0, 0])
                } else {
                    image::Rgba([p.0[0], p.0[1], p.0[2], 255])
                }
            })
        })
        .collect();
    let names = ["pile", "north", "east", "corner"];
    let empty: Vec<&str> = pieces.iter().zip(names).filter(|(p, _)| p.pixels().all(|q| q.0[3] == 0)).map(|(_, n)| n).collect();
    if !empty.is_empty() {
        return Err(tr!(
            format!("Celdas vacías: {}. ¿Es un set de muros (pile, north / east, corner)?", empty.join(", ")),
            format!("Empty cells: {}. Is it a wall set (pile, north / east, corner)?", empty.join(", "))
        ));
    }
    Ok(pieces.iter().map(crate::jobs::anchor_bottom_right).collect())
}

/// The tools, in menu order: on the selected result, or on files picked from disk.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToolCmd {
    RoomSel,
    RoomPng,
    SplitSel,
    SplitPng,
    NoBgSel,
    NoBgPng,
}

impl ToolCmd {
    pub const ALL: [ToolCmd; 6] = [ToolCmd::RoomSel, ToolCmd::RoomPng, ToolCmd::SplitSel, ToolCmd::SplitPng, ToolCmd::NoBgSel, ToolCmd::NoBgPng];

    pub fn needs_selection(self) -> bool {
        matches!(self, ToolCmd::RoomSel | ToolCmd::SplitSel | ToolCmd::NoBgSel)
    }

    pub fn glyph(self) -> &'static str {
        match self {
            ToolCmd::RoomSel => "brick-wall",
            ToolCmd::SplitSel => "scissors",
            ToolCmd::NoBgSel => "eraser",
            _ => "folder-open",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            ToolCmd::RoomSel => tr!("Habitación 3×3 del resultado seleccionado", "3×3 room from the selected result"),
            ToolCmd::RoomPng => tr!("Habitación 3×3 desde un PNG…", "3×3 room from a PNG…"),
            ToolCmd::SplitSel => tr!("Separar el pack seleccionado", "Split the selected pack"),
            ToolCmd::SplitPng => tr!("Separar packs desde PNG…", "Split packs from PNG…"),
            ToolCmd::NoBgSel => tr!("Quitar fondo magenta del seleccionado", "Remove magenta background from selected"),
            ToolCmd::NoBgPng => tr!("Quitar fondo magenta de PNGs…", "Remove magenta background from PNGs…"),
        }
    }

    fn hint(self) -> &'static str {
        match self {
            ToolCmd::RoomPng => tr!("Set 2×2 de 64 px de Dotsmith o de un pack.", "A 2×2 set of 64 px from Dotsmith or a pack."),
            ToolCmd::SplitSel => tr!("Un PNG por celda, en la carpeta de resultados.", "One PNG per cell, in the output folder."),
            ToolCmd::SplitPng => tr!("Pack, set de muros u hoja de sprites.", "Pack, wall set or sprite sheet."),
            ToolCmd::NoBgPng => tr!("Varios a la vez. Los originales no se modifican.", "Several at once. The originals are kept."),
            _ => "",
        }
    }
}

impl Studio {
    pub(super) fn run_tool(&mut self, t: ToolCmd, cx: &mut Context<Self>) {
        self.menu_open = false;
        let sel = self.selected.clone();
        match t {
            ToolCmd::RoomSel => {
                if let Some(p) = sel {
                    self.make_room(&p)
                }
            }
            ToolCmd::SplitSel => {
                if let Some(p) = sel {
                    self.split_pack(&p)
                }
            }
            ToolCmd::NoBgSel => {
                if let Some(p) = sel {
                    self.remove_bg(&[p])
                }
            }
            ToolCmd::RoomPng => {
                let rx = cx.prompt_for_paths(PathPromptOptions {
                    files: true,
                    directories: false,
                    multiple: false,
                    prompt: Some(tr!("Set de muros", "Wall set").into()),
                });
                cx.spawn(async move |this, cx| {
                    if let Ok(Ok(Some(paths))) = rx.await
                        && let Some(p) = paths.into_iter().next()
                    {
                        let _ = this.update(cx, |s, cx| {
                            s.make_room(&p);
                            cx.notify();
                        });
                    }
                })
                .detach();
            }
            ToolCmd::SplitPng => self.pick_pngs(tr!("Packs a separar", "Packs to split"), cx, |s, paths| {
                for p in paths {
                    s.split_pack(&p);
                }
            }),
            ToolCmd::NoBgPng => self.pick_pngs(tr!("Imágenes con fondo magenta", "Images with a magenta background"), cx, |s, paths| s.remove_bg(&paths)),
        }
        cx.notify();
    }

    /// The open Tools menu: a click-away backdrop plus the dropdown under its title-bar button.
    pub(super) fn menu_dropdown(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let has_sel = self.selected.is_some();
        let items = ToolCmd::ALL
            .into_iter()
            .enumerate()
            .flat_map(|(i, t)| {
                let enabled = has_sel || !t.needs_selection();
                let hint = t.hint();
                let sep = (i > 0 && i % 2 == 0).then(|| div().mx(px(10.)).my(px(4.)).h(px(1.)).bg(rgb(BORDER_SOFT())).into_any_element());
                let row = div()
                    .kb(("tool", i))
                    .px(px(10.))
                    .py(px(8.))
                    .flex()
                    .items_start()
                    .gap(px(10.))
                    .rounded(px(7.))
                    .when(enabled, |d| d.cursor_pointer().hover(|s| s.bg(rgb(HOVER()))).on_click(cx.listener(move |this, _, _, cx| this.run_tool(t, cx))))
                    .when(!enabled, |d| d.opacity(0.45))
                    .child(icon(t.glyph()).mt(px(1.)).size(px(16.)).text_color(rgb(ACCENT())))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(div().text_size(px(12.5)).text_color(rgb(TEXT())).child(t.title()))
                            .when(!hint.is_empty(), |d| d.child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(hint))),
                    )
                    .into_any_element();
                sep.into_iter().chain(std::iter::once(row))
            })
            .collect::<Vec<_>>();
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .child(div().id("menu-backdrop").absolute().top_0().left_0().size_full().on_click(cx.listener(|this, _, _, cx| {
                this.menu_open = false;
                cx.notify();
            })))
            .child(
                div()
                    .absolute()
                    .top(px(40.))
                    .right(px(209.))
                    .w(px(360.))
                    .p(px(6.))
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .rounded(px(10.))
                    .bg(rgb(PANEL()))
                    .border_1()
                    .border_color(rgb(BORDER()))
                    .shadow_lg()
                    .child(
                        div()
                            .px(px(10.))
                            .pt(px(6.))
                            .pb(px(4.))
                            .text_size(px(11.))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(rgb(MUTED_2()))
                            .child(tr!("HERRAMIENTAS", "TOOLS")),
                    )
                    .children(items),
            )
            .into_any_element()
    }

    /// File dialog for one or more PNGs, then `then` on the studio with the picked paths.
    fn pick_pngs(&mut self, prompt: &str, cx: &mut Context<Self>, then: impl FnOnce(&mut Studio, Vec<PathBuf>) + 'static) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: true, prompt: Some(prompt.to_string().into()) });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await {
                let _ = this.update(cx, |s, cx| {
                    then(s, paths);
                    cx.notify();
                });
            }
        })
        .detach();
    }

    /// Cut a pack into one PNG per cell, in a folder inside the output folder (never next to a picked file: its
    /// folder may be read-only or someone else's).
    fn split_pack(&mut self, src: &Path) {
        let root = self.model.out_dir.clone();
        if root.as_os_str().is_empty() {
            return self.say(tr!("Espera a que arranque el motor.", "Wait for the engine to start."), Level::Error);
        }
        match crate::pack_split::split(src, &root) {
            Ok(s) => {
                let name = s.dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                self.say(
                    tr!(
                        format!("{} piezas de {}×{} px ({}×{}, {}) en {name}", s.saved, s.cell.0, s.cell.1, s.cols, s.rows, s.how),
                        format!("{} pieces of {}×{} px ({}×{}, {}) in {name}", s.saved, s.cell.0, s.cell.1, s.cols, s.rows, s.how)
                    ),
                    Level::Ok,
                );
                super::reveal_in_explorer(&s.dir);
            }
            Err(e) => self.say(format!("{}: {e}", tr!("Separar pack", "Split pack")), Level::Error),
        }
    }

    /// Remove the flat magenta-ish backdrop of each image and save the result as a new history item (same mode and
    /// settings as the source, so Pack outputs can be split afterwards). The source file is never touched.
    fn remove_bg(&mut self, srcs: &[PathBuf]) {
        let dir = self.model.out_dir.clone();
        if dir.as_os_str().is_empty() {
            return self.say(tr!("Espera a que arranque el motor.", "Wait for the engine to start."), Level::Error);
        }
        let (mut ok, mut last, mut errs) = (0, None, vec![]);
        for src in srcs {
            let name = src.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let res = image::open(src).map_err(|e| e.to_string()).and_then(|im| {
                dotsmith_engine::sprite::remove_magenta_backdrop(&im.to_rgba8(), 60)
                    .ok_or_else(|| tr!("no tiene un fondo magenta liso", "has no flat magenta background").to_string())
            });
            let (clean, key) = match res {
                Ok(r) => r,
                Err(e) => {
                    errs.push(format!("{name}: {e}"));
                    continue;
                }
            };
            let mut json: Value = std::fs::read_to_string(src.with_extension("json"))
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or_else(|| json!({"mode": "nobg", "prompt": ""}));
            json["nobg_of"] = src.to_string_lossy().into();
            json["backdrop"] = format!("#{:02X}{:02X}{:02X}", key[0], key[1], key[2]).into();
            let stem = src.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let png = dir.join(format!("{}_nobg_{stem}.png", crate::jobs::chrono_stamp()));
            let saved = std::fs::create_dir_all(&dir)
                .map_err(|e| e.to_string())
                .and_then(|_| clean.save(&png).map_err(|e| e.to_string()))
                .and_then(|_| std::fs::write(png.with_extension("json"), serde_json::to_string_pretty(&json).unwrap()).map_err(|e| e.to_string()));
            if let Err(e) = saved {
                errs.push(format!("{name}: {e}"));
                continue;
            }
            if let Some(item) = pixels::Item::load(png.clone(), json) {
                self.history.insert(0, item);
            }
            ok += 1;
            last = Some(png);
        }
        if let Some(p) = last {
            self.select(Some(p));
        }
        match (ok, errs.is_empty()) {
            (0, _) => self.say(format!("{}: {}", tr!("Quitar fondo", "Remove background"), errs.join(" · ")), Level::Error),
            (n, true) => self.say(
                tr!(
                    format!("Fondo quitado en {n} imagen{}.", if n == 1 { "" } else { "es" }),
                    format!("Background removed from {n} image{}.", if n == 1 { "" } else { "s" })
                ),
                Level::Ok,
            ),
            (n, false) => self.say(
                tr!(format!("Fondo quitado en {n}; fallaron: {}", errs.join(" · ")), format!("Background removed from {n}; failed: {}", errs.join(" · "))),
                Level::Error,
            ),
        }
    }

    /// Builds the room, saves it in the output folder (never next to the source)
    /// with its .json, adds it to the history and selects it.
    fn make_room(&mut self, src: &Path) {
        let pieces = match wall_pieces(src) {
            Ok(p) => p,
            Err(e) => return self.say(format!("{}: {e}", tr!("Creador de habitación", "Room builder")), Level::Error),
        };
        let dir = self.model.out_dir.clone();
        if dir.as_os_str().is_empty() {
            return self.say(tr!("Espera a que arranque el motor.", "Wait for the engine to start."), Level::Error);
        }
        let room = crate::jobs::wall_room(&pieces, 3);
        let stem = src.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let png: PathBuf = dir.join(format!("{}_walls_room_{stem}.png", crate::jobs::chrono_stamp()));
        let json = json!({"mode": "walls_room", "prompt": "", "room_of": src.to_string_lossy(), "room_tiles": 3, "cell": CELL,
                          "width": room.width(), "height": room.height()});
        let res = std::fs::create_dir_all(&dir)
            .map_err(|e| e.to_string())
            .and_then(|_| room.save(&png).map_err(|e| e.to_string()))
            .and_then(|_| std::fs::write(png.with_extension("json"), serde_json::to_string_pretty(&json).unwrap()).map_err(|e| e.to_string()));
        if let Err(e) = res {
            return self.say(tr!(format!("No se pudo guardar la habitación: {e}"), format!("Could not save the room: {e}")), Level::Error);
        }
        if let Some(item) = pixels::Item::load(png.clone(), json) {
            self.history.insert(0, item);
        }
        self.select(Some(png.clone()));
        let name = png.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        self.say(tr!(format!("Habitación 3×3 guardada: {name}"), format!("3×3 room saved: {name}")), Level::Ok);
    }
}

#[cfg(test)]
mod tests {
    /// The room of a wall sheet, for comparison with a reference build (env ROOM_SRC / ROOM_OUT).
    #[test]
    fn room_matches_training_eval() {
        let (Ok(src), Ok(out)) = (std::env::var("ROOM_SRC"), std::env::var("ROOM_OUT")) else { return };
        let p = super::wall_pieces(std::path::Path::new(&src)).unwrap();
        crate::jobs::wall_room(&p, 3).save(out).unwrap();
    }
}

//! The command palette (Ctrl+K / Ctrl+P): one search box for every mode, setting page, tool and folder, so "how do I
//! get X" is always a few letters away. Arrow keys move, Enter runs, Esc closes.
use gpui::{AnyElement, Context, FontWeight, Window, actions, div, prelude::*, px, rgb};

use super::settings_view::Tab;
use super::tools_menu::ToolCmd;
use super::{MODES, Mode, Studio, kbd};
use crate::icons::icon;
use crate::theme::*;
use crate::tr;

actions!(palette, [PaletteUp, PaletteDown, PaletteRun]);

pub fn bind_keys(cx: &mut gpui::App) {
    use gpui::KeyBinding;
    let c = Some("Palette > TextInput");
    cx.bind_keys([KeyBinding::new("up", PaletteUp, c), KeyBinding::new("down", PaletteDown, c), KeyBinding::new("enter", PaletteRun, c)]);
}

#[derive(Clone, Copy, PartialEq)]
pub enum Cmd {
    Mode(Mode),
    Generate,
    Picker,
    Hub,
    Downloads,
    Settings(Tab),
    Tool(ToolCmd),
    OpenOutput,
    OpenLoras,
    FreeGpu,
}

struct Entry {
    cmd: Cmd,
    glyph: &'static str,
    title: String,
    group: &'static str,
    keys: Option<&'static str>,
}

/// Lower case without Spanish accents, so "angulo" finds "Ángulo".
pub(super) fn fold(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| match c {
            'á' => 'a',
            'é' => 'e',
            'í' => 'i',
            'ó' => 'o',
            'ú' | 'ü' => 'u',
            'ñ' => 'n',
            c => c,
        })
        .collect()
}

impl Studio {
    fn palette_entries(&self) -> Vec<Entry> {
        let mode_group = tr!("Modo", "Mode");
        let go = tr!("Ir a", "Go to");
        let settings = tr!("Ajustes", "Settings");
        let tools = tr!("Herramientas", "Tools");
        let mut out: Vec<Entry> =
            MODES.iter().map(|&m| Entry { cmd: Cmd::Mode(m), glyph: m.glyph(), title: m.title().into(), group: mode_group, keys: None }).collect();
        out.push(Entry { cmd: Cmd::Generate, glyph: "play", title: tr!("Generar", "Generate").into(), group: go, keys: Some("Ctrl ↵") });
        out.push(Entry { cmd: Cmd::Picker, glyph: "box", title: tr!("Elegir modelo y LoRA", "Choose model and LoRA").into(), group: go, keys: Some("Ctrl L") });
        out.push(Entry {
            cmd: Cmd::Hub,
            glyph: "download",
            title: tr!("Descargar o gestionar modelos", "Download or manage models").into(),
            group: go,
            keys: None,
        });
        out.push(Entry { cmd: Cmd::Downloads, glyph: "download", title: tr!("Descargas en curso", "Downloads in progress").into(), group: go, keys: None });
        out.push(Entry {
            cmd: Cmd::OpenOutput,
            glyph: "folder-open",
            title: tr!("Abrir la carpeta de resultados", "Open the output folder").into(),
            group: go,
            keys: None,
        });
        out.push(Entry {
            cmd: Cmd::OpenLoras,
            glyph: "folder-open",
            title: tr!("Abrir la carpeta de LoRAs", "Open the LoRA folder").into(),
            group: go,
            keys: None,
        });
        if self.model.loaded.is_some() {
            out.push(Entry { cmd: Cmd::FreeGpu, glyph: "cpu", title: tr!("Liberar la GPU", "Free the GPU").into(), group: go, keys: None });
        }
        for t in Tab::ALL {
            out.push(Entry {
                cmd: Cmd::Settings(t),
                glyph: t.glyph(),
                title: t.title().into(),
                group: settings,
                keys: (t == Tab::General).then_some("Ctrl ,"),
            });
        }
        let has_sel = self.selected.is_some();
        for t in ToolCmd::ALL.into_iter().filter(|t| has_sel || !t.needs_selection()) {
            out.push(Entry { cmd: Cmd::Tool(t), glyph: t.glyph(), title: t.title().into(), group: tools, keys: None });
        }
        out
    }

    fn palette_matches(&self, cx: &Context<Self>) -> Vec<Entry> {
        let q = fold(&self.palette_query.read(cx).text());
        let words: Vec<&str> = q.split_whitespace().collect();
        self.palette_entries()
            .into_iter()
            .filter(|e| {
                let hay = fold(&format!("{} {}", e.title, e.group));
                words.iter().all(|w| hay.contains(w))
            })
            .collect()
    }

    pub(super) fn open_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.menu_open = false;
        self.palette_open = true;
        self.palette_sel = 0;
        self.palette_query.update(cx, |t, cx| t.set_text("", cx));
        window.focus(&self.palette_query.read(cx).focus_handle.clone());
        cx.notify();
    }

    pub(super) fn close_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.palette_open = false;
        window.focus(&self.focus);
        cx.notify();
    }

    fn run_command(&mut self, cmd: Cmd, window: &mut Window, cx: &mut Context<Self>) {
        self.close_palette(window, cx);
        match cmd {
            Cmd::Mode(m) => self.mode = m,
            Cmd::Generate => self.start(cx),
            Cmd::Picker => self.open_model_picker(),
            Cmd::Hub => self.open_hub(None, cx),
            Cmd::Downloads => self.downloads_open = true,
            Cmd::Settings(t) => {
                self.settings_tab = t;
                self.settings_open = true;
            }
            Cmd::Tool(t) => self.run_tool(t, cx),
            Cmd::OpenOutput => {
                let dir = self.model.out_dir.clone();
                let _ = std::fs::create_dir_all(&dir);
                cx.open_with_system(&dir);
            }
            Cmd::OpenLoras => {
                if let Some(dir) = self.settings.lora_library() {
                    let _ = std::fs::create_dir_all(&dir);
                    cx.open_with_system(&dir);
                }
            }
            Cmd::FreeGpu => self.send(serde_json::json!({"cmd": "unload"})),
        }
        cx.notify();
    }

    /// Keep the selected command in view; the first one also brings its group title.
    fn palette_follow(&self) {
        if let Some(&ix) = self.palette_rows.get(self.palette_sel) {
            self.palette_scroll.scroll_to_item(if self.palette_sel == 0 { 0 } else { ix });
        }
    }

    pub(super) fn palette_view(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let query = self.palette_query.read(cx).text();
        if query != self.palette_last {
            self.palette_last = query;
            self.palette_sel = 0;
        }
        let matches = self.palette_matches(cx);
        let sel = self.palette_sel.min(matches.len().saturating_sub(1));
        let mut last_group = "";
        let mut rows: Vec<AnyElement> = vec![];
        self.palette_rows.clear();
        for (i, e) in matches.iter().enumerate() {
            if e.group != last_group {
                last_group = e.group;
                rows.push(
                    div()
                        .flex_none()
                        .px(px(12.))
                        .pt(px(if i == 0 { 4. } else { 10. }))
                        .pb(px(4.))
                        .text_size(px(10.5))
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(rgb(MUTED_2()))
                        .child(e.group.to_uppercase())
                        .into_any_element(),
                );
            }
            let on = i == sel;
            let cmd = e.cmd;
            self.palette_rows.push(rows.len());
            rows.push(
                div()
                    .id(("pal", i))
                    .flex_none()
                    .h(px(34.))
                    .px(px(12.))
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .rounded(px(7.))
                    .cursor_pointer()
                    .when(on, |d| d.bg(rgb(RAISED())))
                    .when(!on, |d| d.hover(|s| s.bg(rgb(PANEL()))))
                    .child(icon(e.glyph).size(px(15.)).text_color(rgb(if on { ACCENT() } else { MUTED() })))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .overflow_hidden()
                            .text_ellipsis()
                            .whitespace_nowrap()
                            .text_size(px(13.))
                            .text_color(rgb(TEXT()))
                            .child(e.title.clone()),
                    )
                    .children(e.keys.map(kbd))
                    .on_click(cx.listener(move |this, _, w, cx| this.run_command(cmd, w, cx)))
                    .into_any_element(),
            );
        }
        let empty = matches.is_empty();
        let card = div()
            .id("palette")
            .occlude()
            .w(px(600.))
            .max_w_full()
            .max_h(px(460.))
            .rounded(px(12.))
            .bg(rgb(BG()))
            .border_1()
            .border_color(rgb(BORDER()))
            .shadow_lg()
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(
                div()
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .px(px(16.))
                    .py(px(8.))
                    .border_b_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .child(icon("search").size(px(16.)).text_color(rgb(MUTED())))
                    .child(div().flex_1().child(self.palette_query.clone()))
                    .child(kbd("Esc")),
            )
            .child(
                div()
                    .id("palette-list")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.palette_scroll)
                    .p(px(6.))
                    .flex()
                    .flex_col()
                    .gap(px(1.))
                    .children(rows)
                    .when(empty, |d| {
                        d.child(
                            div()
                                .py(px(28.))
                                .flex()
                                .justify_center()
                                .text_size(px(12.5))
                                .text_color(rgb(MUTED_2()))
                                .child(tr!("Nada coincide", "Nothing matches")),
                        )
                    }),
            )
            .child(
                div()
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .px(px(14.))
                    .py(px(8.))
                    .border_t_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .bg(rgb(PANEL()))
                    .child(kbd("↑↓"))
                    .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).mr(px(8.)).child(tr!("moverse", "move")))
                    .child(kbd("↵"))
                    .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(tr!("abrir", "open"))),
            );
        let n = matches.len();
        let layer =
            self.modal_layer("palette-veil", div().size_full().flex().flex_col().items_center().pt(px(70.)).child(card), |s, w, cx| s.close_palette(w, cx), cx);
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .key_context("Palette")
            .on_action(cx.listener(move |this, _: &PaletteUp, _, cx| {
                this.palette_sel = (this.palette_sel + n.max(1) - 1) % n.max(1);
                this.palette_follow();
                cx.notify();
            }))
            .on_action(cx.listener(move |this, _: &PaletteDown, _, cx| {
                this.palette_sel = (this.palette_sel + 1) % n.max(1);
                this.palette_follow();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &PaletteRun, w, cx| {
                let m = this.palette_matches(cx);
                if let Some(e) = m.get(this.palette_sel.min(m.len().saturating_sub(1))) {
                    let cmd = e.cmd;
                    this.run_command(cmd, w, cx);
                }
            }))
            .child(layer)
            .into_any_element()
    }
}

//! The "Animar" mode: a sprite from the history or a file gets an ambient loop (crate::item_anim), previewed live and
//! saved as a one-row sheet that the creature player plays and exports. No model involved.
//! The loop is a stack of layers, one effect each; the open layer shows its options and an editor over the sprite to
//! paint its area, pick the colour it acts on, or set where it starts.
use gpui::{
    AnyElement, Bounds, Context, Div, FontWeight, MouseButton, MouseDownEvent, MouseMoveEvent, Pixels, Point, Stateful, canvas, div, fill, point, prelude::*,
    px, rgb, rgba, size,
};
use image::RgbaImage;
use serde_json::{Value, json};
use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use super::{Level, Studio, chip, icon, icon_btn, label, paint_checker, paint_pixels, section, tooltip};
use crate::item_anim::{Area, ELEMENTS, Effect, Layer, Settings, animate, area_mask, hsv, rgb_of, strip};
use crate::pixels;
use crate::theme::*;
use crate::tr;

const FRAMES: [usize; 6] = [4, 6, 8, 12, 16, 24];
pub const FPS: [u32; 5] = [4, 6, 8, 10, 12];
/// Sources above this are regular images, not sprites: the effects are drawn one sprite pixel at a time.
const MAX_SIDE: u32 = 256;
const EXTRA_COLORS: [[u8; 3]; 2] = [[200, 30, 40], [240, 240, 240]];

/// What a click on the layer editor does.
#[derive(Clone, Copy, PartialEq)]
enum Tool {
    Paint,
    Erase,
    Pick,
    Origin,
}

/// name, hint, shown value, change by -1 / +1
type AdvancedRow = (&'static str, &'static str, String, fn(&mut Settings, i32));
type Painted = Rc<Cell<Option<Bounds<Pixels>>>>;

/// A layer's area as pixels, for the editor's tint, with what it was computed from.
struct Tint {
    source: PathBuf,
    area: Area,
    effect: Effect,
    threshold: u32,
    pixels: Arc<Vec<bool>>,
}

pub struct AnimateState {
    pub settings: Settings,
    pub element: Option<usize>,
    pub fps: u32,
    /// the layer whose options are open
    open: Option<usize>,
    /// the list of effects to add is showing
    adding: bool,
    tool: Tool,
    preview: Option<(PathBuf, Settings, Arc<Vec<RgbaImage>>)>,
    source: Option<(PathBuf, Arc<RgbaImage>)>,
    /// the sprite the painted areas and origins belong to: another source drops them
    placed_on: Option<PathBuf>,
    tint: Option<Tint>,
    editor: Painted,
    /// the colour picker: open, its hue / saturation / brightness, where its parts were painted, the hex last applied
    color_open: bool,
    hsv: (f32, f32, f32),
    sv_box: Painted,
    hue_bar: Painted,
    hex_seen: String,
}

impl Default for AnimateState {
    fn default() -> Self {
        AnimateState {
            settings: Settings::default(),
            element: None,
            fps: 10,
            open: None,
            adding: false,
            tool: Tool::Paint,
            preview: None,
            source: None,
            placed_on: None,
            tint: None,
            editor: Rc::default(),
            color_open: false,
            hsv: (0.08, 0.9, 0.95),
            sv_box: Rc::default(),
            hue_bar: Rc::default(),
            hex_seen: String::new(),
        }
    }
}

fn effect_label(e: Effect) -> &'static str {
    match e {
        Effect::Pulse => tr!("Pulso", "Pulse"),
        Effect::Glint => tr!("Destello", "Glint"),
        Effect::Twinkle => tr!("Titilar", "Twinkle"),
        Effect::Shine => tr!("Reflejo", "Shine"),
        Effect::Motes => tr!("Chispas", "Sparks"),
        Effect::Plasma => "Plasma",
        Effect::Shock => tr!("Onda de choque", "Shockwave"),
        Effect::Trail => tr!("Estela", "Trail"),
        Effect::Cycle => tr!("Vetas", "Veins"),
        Effect::Rim => tr!("Aura", "Aura"),
        Effect::Blink => tr!("Parpadeo", "Blink"),
        Effect::Hue => tr!("Tono", "Hue"),
        Effect::Wave => tr!("Ondear", "Wave"),
        Effect::Beat => tr!("Latido", "Heartbeat"),
        Effect::Heat => tr!("Calor", "Heat"),
        Effect::Flames => tr!("Llamas", "Flames"),
        Effect::Frost => tr!("Escarcha", "Frost"),
        Effect::Icicles => tr!("Carámbanos", "Icicles"),
        Effect::Snow => tr!("Nieve", "Snow"),
        Effect::Arcs => tr!("Rayos", "Arcs"),
        Effect::Drip => tr!("Goteo", "Drip"),
        Effect::Bubbles => tr!("Burbujas", "Bubbles"),
        Effect::Smoke => tr!("Humo", "Smoke"),
        Effect::Orbit => tr!("Órbita", "Orbit"),
    }
}

fn effect_icon(e: Effect) -> &'static str {
    match e {
        Effect::Pulse => "activity",
        Effect::Glint => "star",
        Effect::Twinkle => "sparkles",
        Effect::Shine => "scan-line",
        Effect::Motes => "sparkle",
        Effect::Plasma => "radio",
        Effect::Shock => "target",
        Effect::Trail => "route",
        Effect::Cycle => "git-branch",
        Effect::Rim => "circle-dashed",
        Effect::Blink => "eye",
        Effect::Hue => "palette",
        Effect::Wave => "waves",
        Effect::Beat => "heart-pulse",
        Effect::Heat => "thermometer",
        Effect::Flames => "flame",
        Effect::Frost => "snowflake",
        Effect::Icicles => "chevrons-down",
        Effect::Snow => "cloud-snow",
        Effect::Arcs => "zap",
        Effect::Drip => "droplet",
        Effect::Bubbles => "flask-conical",
        Effect::Smoke => "cloud",
        Effect::Orbit => "orbit",
    }
}

/// The effects by what they do, the order of the "add" list.
const GROUPS: [&[Effect]; 6] = [
    &[Effect::Pulse, Effect::Glint, Effect::Twinkle, Effect::Shine, Effect::Rim, Effect::Blink, Effect::Hue],
    &[Effect::Plasma, Effect::Shock, Effect::Trail, Effect::Cycle, Effect::Arcs, Effect::Orbit],
    &[Effect::Motes, Effect::Bubbles, Effect::Smoke, Effect::Drip],
    &[Effect::Heat, Effect::Flames],
    &[Effect::Frost, Effect::Icicles, Effect::Snow],
    &[Effect::Wave, Effect::Beat],
];

fn group_label(i: usize) -> &'static str {
    match i {
        0 => tr!("Luz", "Light"),
        1 => tr!("Energía", "Energy"),
        2 => tr!("Partículas", "Particles"),
        3 => tr!("Fuego", "Fire"),
        4 => tr!("Hielo", "Ice"),
        _ => tr!("Movimiento", "Motion"),
    }
}

const ELEMENT_ICONS: [&str; 6] = ["flame", "snowflake", "zap", "leaf", "skull", "sun"];

/// A square button that is only an icon, lit while `active`.
fn icon_toggle(id: impl Into<gpui::ElementId>, name: &'static str, active: bool) -> Stateful<Div> {
    div()
        .id(id.into())
        .w(px(30.))
        .h(px(30.))
        .flex()
        .items_center()
        .justify_center()
        .rounded(px(7.))
        .cursor_pointer()
        .when(active, |d| d.bg(rgb(ACCENT())))
        .when(!active, |d| d.bg(rgb(RAISED())).border_1().border_color(rgb(BORDER())).hover(|s| s.bg(rgb(HOVER()))))
        .child(icon(name).size(px(15.)).text_color(rgb(if active { ACCENT_INK() } else { TEXT() })))
}

/// What sets a layer apart from the effect's defaults, for its row: "Painted · ×2 · 50 %".
fn layer_summary(l: &Layer) -> String {
    let mut parts = vec![];
    if !matches!(l.area, Area::Auto) {
        parts.push(area_label(area_index(&l.area)).to_string());
    }
    if l.repeats > 1 {
        parts.push(format!("×{}", l.repeats));
    }
    if l.strength < 0.99 {
        parts.push(format!("{:.0} %", l.strength * 100.));
    }
    parts.join(" · ")
}

fn amount_label(e: Effect) -> &'static str {
    match e {
        Effect::Twinkle => tr!("Destellos", "Glints"),
        Effect::Wave => tr!("Amplitud (px)", "Amplitude (px)"),
        Effect::Shock => tr!("Empuje (px)", "Push (px)"),
        Effect::Beat => tr!("Hinchado (px)", "Swell (px)"),
        Effect::Orbit => tr!("Orbes", "Orbs"),
        Effect::Flames => tr!("Altura (px)", "Height (px)"),
        Effect::Frost => tr!("Profundidad (px)", "Depth (px)"),
        Effect::Icicles => tr!("Carámbanos", "Icicles"),
        Effect::Snow => tr!("Copos", "Flakes"),
        _ => tr!("Chispas", "Sparks"),
    }
}

fn element_label(i: usize) -> &'static str {
    match i {
        0 => tr!("Fuego", "Fire"),
        1 => tr!("Hielo", "Ice"),
        2 => tr!("Energía", "Energy"),
        3 => tr!("Tierra", "Earth"),
        4 => tr!("Muerte", "Death"),
        _ => tr!("Sagrado", "Holy"),
    }
}

fn area_label(i: usize) -> &'static str {
    match i {
        0 => "Auto",
        1 => tr!("Todo", "All"),
        2 => tr!("Brillo", "Glow"),
        3 => "Color",
        _ => tr!("Pintada", "Painted"),
    }
}

fn area_index(a: &Area) -> usize {
    match a {
        Area::Auto => 0,
        Area::All => 1,
        Area::Glow => 2,
        Area::Color(_) => 3,
        Area::Mask(_) => 4,
    }
}

fn hex(c: [u8; 3]) -> u32 {
    (c[0] as u32) << 16 | (c[1] as u32) << 8 | c[2] as u32
}

fn rgb3(v: &Value) -> Option<[u8; 3]> {
    let c = v.as_array()?;
    Some([c.first()?.as_u64()? as u8, c.get(1)?.as_u64()? as u8, c.get(2)?.as_u64()? as u8])
}

fn layer_json(l: &Layer) -> Value {
    let area = match &l.area {
        Area::Auto => json!("auto"),
        Area::All => json!("all"),
        Area::Glow => json!("glow"),
        Area::Color(c) => json!({ "color": c }),
        Area::Mask(m) => json!({"mask": m.iter().map(|&b| if b { '1' } else { '0' }).collect::<String>()}),
    };
    json!({"effect": l.effect.key(), "color": l.color, "strength": l.strength, "repeats": l.repeats, "area": area, "origin": l.origin, "amount": l.amount,
        "on": l.on})
}

fn layer_of(v: &Value) -> Option<Layer> {
    let mut l = Layer::new(Effect::from_key(v["effect"].as_str()?)?);
    l.color = rgb3(&v["color"]);
    l.strength = v["strength"].as_f64().map_or(l.strength, |s| s as f32);
    l.repeats = v["repeats"].as_u64().map_or(l.repeats, |r| r as usize);
    l.amount = v["amount"].as_u64().map_or(l.amount, |a| a as usize);
    l.on = v["on"].as_bool().unwrap_or(true);
    l.origin = v["origin"].as_array().and_then(|o| Some((o.first()?.as_u64()? as u32, o.get(1)?.as_u64()? as u32)));
    l.area = match &v["area"] {
        Value::String(s) if s == "all" => Area::All,
        Value::String(s) if s == "glow" => Area::Glow,
        a if a["color"].is_array() => rgb3(&a["color"]).map_or(Area::Auto, Area::Color),
        a if a["mask"].is_string() => Area::Mask(a["mask"].as_str().unwrap_or("").chars().map(|c| c == '1').collect()),
        _ => Area::Auto,
    };
    Some(l)
}

impl Studio {
    fn animate_source(&self) -> Option<(PathBuf, u32, u32)> {
        self.source.as_ref().map(|s| (s.path.clone(), s.w, s.h))
    }

    /// The source sprite, read once per file. A different sprite drops what was placed on the previous one.
    fn animate_image(&mut self) -> Option<Arc<RgbaImage>> {
        let (path, ..) = self.animate_source()?;
        if let Some((p, im)) = &self.animate.source
            && *p == path
        {
            return Some(im.clone());
        }
        let im = Arc::new(image::open(&path).ok()?.to_rgba8());
        if self.animate.placed_on.as_ref() != Some(&path) {
            for l in &mut self.animate.settings.layers {
                l.origin = None;
                if matches!(l.area, Area::Mask(_)) {
                    l.area = Area::Auto;
                }
            }
        }
        self.animate.placed_on = Some(path.clone());
        self.animate.source = Some((path, im.clone()));
        Some(im)
    }

    /// The loop for the current source and options, recomputed only when either changes.
    pub(super) fn animate_frames(&mut self) -> Option<Arc<Vec<RgbaImage>>> {
        let (path, w, h) = self.animate_source()?;
        if w.max(h) > MAX_SIDE {
            return None;
        }
        let im = self.animate_image()?;
        if let Some((p, ps, f)) = &self.animate.preview
            && *p == path
            && *ps == self.animate.settings
        {
            return Some(f.clone());
        }
        let s = self.animate.settings.clone();
        let frames = Arc::new(animate(&im, &s)?);
        self.animate.preview = Some((path, s, frames.clone()));
        Some(frames)
    }

    /// Frame shown by the viewer: the live loop at the player's position.
    pub(super) fn animate_frame(&mut self) -> Option<Arc<RgbaImage>> {
        let frames = self.animate_frames()?;
        Some(Arc::new(frames[self.anim.frame % frames.len()].clone()))
    }

    /// An option changed: the viewer goes back to the live preview at the chosen speed.
    fn animate_changed(&mut self, cx: &mut Context<Self>) {
        self.select(None);
        self.anim.fps = self.animate.fps;
        self.anim.playing = true;
        cx.notify();
    }

    /// Changes layer `i` and drops the element preset it no longer is.
    fn edit_layer(&mut self, i: usize, change: impl FnOnce(&mut Layer), cx: &mut Context<Self>) {
        if let Some(l) = self.animate.settings.layers.get_mut(i) {
            change(l);
            self.animate.element = None;
            self.animate_changed(cx);
        }
    }

    pub(super) fn save_animation(&mut self, cx: &mut Context<Self>) {
        let dir = self.model.out_dir.clone();
        let Some((source, w, h)) = self.animate_source() else {
            self.say(tr!("Carga primero un sprite (abrir, arrastrar o Ctrl+V).", "Load a sprite first (open, drag or Ctrl+V)."), Level::Error);
            cx.notify();
            return;
        };
        let Some(frames) = self.animate_frames() else {
            self.say(tr!("Este origen no se puede animar.", "This source can't be animated."), Level::Error);
            cx.notify();
            return;
        };
        if dir.as_os_str().is_empty() {
            self.say(tr!("Espera a que arranque el motor.", "Wait for the engine to start."), Level::Error);
            cx.notify();
            return;
        }
        let a = &self.animate;
        let s = &a.settings;
        let png = dir.join(format!("{}_animate_{}.png", crate::jobs::chrono_stamp(), s.seed));
        let json = json!({"mode": "animate", "prompt": "", "source": source.to_string_lossy(), "frames": frames.len(), "cell": w, "cell_h": h,
            "dirs": ["item"], "fps": a.fps, "layers": s.layers.iter().map(layer_json).collect::<Vec<_>>(),
            "element": a.element.map(|i| ELEMENTS[i].key), "threshold": s.threshold, "seed": s.seed,
            "width": w, "height": h, "app": format!("Dotsmith {}", env!("CARGO_PKG_VERSION"))});
        let sheet = strip(&frames);
        let res = std::fs::create_dir_all(&dir)
            .map_err(|e| e.to_string())
            .and_then(|_| sheet.save(&png).map_err(|e| e.to_string()))
            .and_then(|_| std::fs::write(png.with_extension("json"), serde_json::to_string_pretty(&json).unwrap()).map_err(|e| e.to_string()));
        if let Err(e) = res {
            self.say(tr!(format!("No se pudo guardar la animación: {e}"), format!("Could not save the animation: {e}")), Level::Error);
            cx.notify();
            return;
        }
        if let Some(item) = pixels::Item::load(png.clone(), json) {
            self.history.insert(0, item);
        }
        self.select(Some(png.clone()));
        let name = png.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        self.say(tr!(format!("Animación guardada: {name}"), format!("Animation saved: {name}")), Level::Info);
        cx.notify();
    }

    /// A saved animation's options back into the form.
    pub(super) fn restore_animation(&mut self, j: &Value) {
        let s = &mut self.animate.settings;
        if let Some(layers) = j["layers"].as_array() {
            s.layers = layers.iter().filter_map(layer_of).collect();
        } else if let Some(fx) = j["effects"].as_array() {
            // saved before layers: one colour for every effect
            let color = rgb3(&j["color"]);
            s.layers = fx.iter().filter_map(|v| v.as_str().and_then(Effect::from_key)).map(|e| Layer { color, ..Layer::new(e) }).collect();
        }
        s.frames = j["frames"].as_u64().map_or(s.frames, |v| v as usize);
        s.threshold = j["threshold"].as_f64().map_or(s.threshold, |v| v as f32);
        s.seed = j["seed"].as_u64().unwrap_or(s.seed);
        self.animate.element = j["element"].as_str().and_then(|k| ELEMENTS.iter().position(|e| e.key == k));
        self.animate.fps = j["fps"].as_u64().map_or(self.animate.fps, |v| v as u32);
        self.animate.placed_on = j["source"].as_str().map(PathBuf::from);
        self.animate.open = None;
    }

    /// The sprite pixel at `pos`, laid out like paint_pixels (fitted and centred in `b`).
    fn picked_pixel(b: Bounds<Pixels>, iw: u32, ih: u32, pos: Point<Pixels>) -> Option<(u32, u32)> {
        let (bw, bh) = (f32::from(b.size.width), f32::from(b.size.height));
        let cell = (bw / iw as f32).min(bh / ih as f32);
        let (ox, oy) = (f32::from(b.origin.x) + (bw - cell * iw as f32) / 2., f32::from(b.origin.y) + (bh - cell * ih as f32) / 2.);
        let (x, y) = ((f32::from(pos.x) - ox) / cell, (f32::from(pos.y) - oy) / cell);
        (x >= 0. && y >= 0. && (x as u32) < iw && (y as u32) < ih).then_some((x as u32, y as u32))
    }

    /// The pixels layer `i` acts on, recomputed when its area or the sprite changes.
    fn layer_tint(&mut self, i: usize) -> Option<Arc<Vec<bool>>> {
        let im = self.animate_image()?;
        let (path, ..) = self.animate_source()?;
        let l = self.animate.settings.layers.get(i)?;
        let thr = self.animate.settings.threshold.to_bits();
        if let Some(t) = &self.animate.tint
            && t.source == path
            && t.area == l.area
            && t.effect == l.effect
            && t.threshold == thr
        {
            return Some(t.pixels.clone());
        }
        let pixels = Arc::new(area_mask(&im, &l.area, l.effect, self.animate.settings.threshold));
        self.animate.tint = Some(Tint { source: path, area: l.area.clone(), effect: l.effect, threshold: thr, pixels: pixels.clone() });
        Some(pixels)
    }

    /// The editor tools layer `i` has, and the one in use.
    fn layer_tools(&self, i: usize) -> (Vec<Tool>, Option<Tool>) {
        let Some(l) = self.animate.settings.layers.get(i) else { return (vec![], None) };
        let mut tools = match l.area {
            Area::Mask(_) => vec![Tool::Paint, Tool::Erase],
            Area::Color(_) => vec![Tool::Pick],
            _ => vec![],
        };
        if l.effect.has_origin() {
            tools.push(Tool::Origin);
        }
        let current = if tools.contains(&self.animate.tool) { Some(self.animate.tool) } else { tools.first().copied() };
        (tools, current)
    }

    /// A click or drag on the editor of layer `i`.
    fn editor_input(&mut self, i: usize, pos: Point<Pixels>, drag: bool, cx: &mut Context<Self>) {
        let (Some(im), Some(b), (_, Some(tool))) = (self.animate_image(), self.animate.editor.get(), self.layer_tools(i)) else { return };
        let (w, h) = im.dimensions();
        let Some((x, y)) = Self::picked_pixel(b, w, h, pos) else { return };
        match tool {
            Tool::Paint | Tool::Erase => {
                let r = (w.max(h) / 48) as i32;
                self.edit_layer(
                    i,
                    |l| {
                        if let Area::Mask(m) = &mut l.area {
                            m.resize((w * h) as usize, false);
                            for yy in (y as i32 - r).max(0)..=(y as i32 + r).min(h as i32 - 1) {
                                for xx in (x as i32 - r).max(0)..=(x as i32 + r).min(w as i32 - 1) {
                                    m[(yy as u32 * w + xx as u32) as usize] = tool == Tool::Paint;
                                }
                            }
                        }
                    },
                    cx,
                );
            }
            Tool::Pick if !drag => {
                let p = im.get_pixel(x, y).0;
                if p[3] > 127 {
                    self.edit_layer(i, |l| l.area = Area::Color([p[0], p[1], p[2]]), cx);
                }
            }
            Tool::Origin if !drag => self.edit_layer(i, |l| l.origin = Some((x, y)), cx),
            _ => {}
        }
    }

    /// The sprite with the layer's area tinted and its origin ringed; clicks go to the tool in use.
    fn layer_editor(&mut self, i: usize, cx: &mut Context<Self>) -> Option<AnyElement> {
        let im = self.animate_image()?;
        let tint = self.layer_tint(i)?;
        let origin = self.animate.settings.layers.get(i)?.origin;
        let (tools, tool) = self.layer_tools(i);
        let (bounds, scale) = (self.animate.editor.clone(), self.scale_factor);
        let (iw, ih) = im.dimensions();
        let look = |t: Tool| match t {
            Tool::Paint => ("brush", tr!("Pintar", "Paint")),
            Tool::Erase => ("eraser", tr!("Borrar", "Erase")),
            Tool::Pick => ("pipette", tr!("Cuentagotas: clic en un color del sprite", "Eyedropper: click a colour on the sprite")),
            Tool::Origin => ("crosshair", tr!("Origen: clic donde empieza", "Origin: click where it starts")),
        };
        let side = div()
            .flex()
            .flex_col()
            .gap(px(6.))
            .children(tools.into_iter().enumerate().map(|(k, t)| {
                let (name, tip) = look(t);
                icon_toggle(("tool", k), name, tool == Some(t)).tooltip(tooltip(tip)).on_click(cx.listener(move |this, _, _, cx| {
                    this.animate.tool = t;
                    cx.notify();
                }))
            }))
            .when(origin.is_some(), |d| {
                d.child(
                    icon_toggle("origin-auto", "rotate-ccw", false)
                        .tooltip(tooltip(tr!("Origen automático", "Automatic origin")))
                        .on_click(cx.listener(move |this, _, _, cx| this.edit_layer(i, |l| l.origin = None, cx))),
                )
            });
        Some(
            div()
                .flex()
                .gap(px(8.))
                .child(
                    div()
                        .id("layer-editor")
                        .relative()
                        .w(px(200.))
                        .h(px(200.))
                        .rounded(px(6.))
                        .overflow_hidden()
                        .border_1()
                        .border_color(rgb(BORDER()))
                        .cursor_crosshair()
                        .child(canvas(|_, _, _| (), |b, _, window, _| paint_checker(b, 10., window)).absolute().size_full())
                        .child(
                            canvas(
                                move |b, _, _| bounds.set(Some(b)),
                                move |b, _, window, _| {
                                    paint_pixels(b, &im, scale, window);
                                    let (bw, bh) = (f32::from(b.size.width), f32::from(b.size.height));
                                    let c = (bw / iw as f32).min(bh / ih as f32);
                                    let at = |x: f32, y: f32| {
                                        point(b.origin.x + px((bw - c * iw as f32) / 2. + c * x), b.origin.y + px((bh - c * ih as f32) / 2. + c * y))
                                    };
                                    for y in 0..ih {
                                        let mut x = 0;
                                        while x < iw {
                                            if !tint[(y * iw + x) as usize] {
                                                x += 1;
                                                continue;
                                            }
                                            let start = x;
                                            while x < iw && tint[(y * iw + x) as usize] {
                                                x += 1;
                                            }
                                            window.paint_quad(fill(
                                                Bounds::new(at(start as f32, y as f32), size(px(c * (x - start) as f32), px(c))),
                                                rgba(0xeee2cb66),
                                            ));
                                        }
                                    }
                                    if let Some((x, y)) = origin {
                                        let ring = px(c * 3.);
                                        window.paint_quad(
                                            fill(Bounds::new(at(x as f32 - 1., y as f32 - 1.), size(ring, ring)), gpui::transparent_black())
                                                .border_widths(px(2.))
                                                .border_color(rgb(ACCENT())),
                                        );
                                    }
                                },
                            )
                            .absolute()
                            .size_full(),
                        )
                        .on_mouse_down(MouseButton::Left, cx.listener(move |this, ev: &MouseDownEvent, _, cx| this.editor_input(i, ev.position, false, cx)))
                        .on_mouse_move(cx.listener(move |this, ev: &MouseMoveEvent, _, cx| {
                            if ev.pressed_button == Some(MouseButton::Left) {
                                this.editor_input(i, ev.position, true, cx);
                            }
                        })),
                )
                .child(side)
                .into_any_element(),
        )
    }

    /// Sets the open layer's colour from the picker's hue / saturation / brightness and shows it in the hex field.
    fn pick_hsv(&mut self, h: f32, s: f32, v: f32, cx: &mut Context<Self>) {
        self.animate.hsv = (h.clamp(0., 0.999), s.clamp(0., 1.), v.clamp(0., 1.));
        let (h, s, v) = self.animate.hsv;
        let c = rgb_of(h, s, v).map(|x| x.round() as u8);
        let text = format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2]);
        self.animate.hex_seen = text.clone();
        self.animate_hex.update(cx, |t, cx| t.set_text(&text, cx));
        if let Some(i) = self.animate.open {
            self.edit_layer(i, |l| l.color = Some(c), cx);
        }
    }

    /// A small picker: saturation x brightness square, hue bar, and a #RRGGBB field.
    fn color_picker(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let typed = self.animate_hex.read(cx).text().trim().trim_start_matches('#').to_uppercase();
        if typed != self.animate.hex_seen.trim_start_matches('#')
            && typed.len() == 6
            && let Ok(v) = u32::from_str_radix(&typed, 16)
        {
            let c = [(v >> 16) as u8, (v >> 8) as u8, v as u8];
            self.animate.hsv = hsv(c.map(|x| x as f32));
            self.animate.hex_seen = format!("#{typed}");
            if let Some(l) = self.animate.open.and_then(|i| self.animate.settings.layers.get_mut(i)) {
                l.color = Some(c);
            }
            self.select(None);
        }
        let (h, s, v) = self.animate.hsv;
        let (sv, hue) = (self.animate.sv_box.clone(), self.animate.hue_bar.clone());
        fn at(b: Bounds<Pixels>, p: Point<Pixels>) -> (f32, f32) {
            (f32::from(p.x - b.origin.x) / f32::from(b.size.width), f32::from(p.y - b.origin.y) / f32::from(b.size.height))
        }
        fn on_sv(this: &mut Studio, p: Point<Pixels>, cx: &mut Context<Studio>) {
            if let Some(b) = this.animate.sv_box.get() {
                let (x, y) = at(b, p);
                let h = this.animate.hsv.0;
                this.pick_hsv(h, x, 1. - y, cx);
            }
        }
        fn on_hue(this: &mut Studio, p: Point<Pixels>, cx: &mut Context<Studio>) {
            if let Some(b) = this.animate.hue_bar.get() {
                let (x, _) = at(b, p);
                let (_, s, v) = this.animate.hsv;
                this.pick_hsv(x, s, v, cx);
            }
        }
        let swatch = |c: [f32; 3]| rgb(hex(c.map(|x| x.round() as u8)));
        let paint_sv = move |b: Bounds<Pixels>, window: &mut gpui::Window| {
            let (cols, rows) = (32, 16);
            let (cw, ch) = (b.size.width / cols as f32, b.size.height / rows as f32);
            for r in 0..rows {
                for c in 0..cols {
                    let col = rgb_of(h, c as f32 / (cols - 1) as f32, 1. - r as f32 / (rows - 1) as f32);
                    let o = point(b.origin.x + cw * c as f32, b.origin.y + ch * r as f32);
                    window.paint_quad(fill(Bounds::new(o, size(cw + px(0.5), ch + px(0.5))), swatch(col)));
                }
            }
            let m = point(b.origin.x + b.size.width * s - px(4.), b.origin.y + b.size.height * (1. - v) - px(4.));
            window.paint_quad(fill(Bounds::new(m, size(px(8.), px(8.))), gpui::transparent_black()).border_widths(px(2.)).border_color(rgb(0xffffff)));
        };
        let paint_hue = move |b: Bounds<Pixels>, window: &mut gpui::Window| {
            let n = 36;
            let cw = b.size.width / n as f32;
            for i in 0..n {
                let o = point(b.origin.x + cw * i as f32, b.origin.y);
                window.paint_quad(fill(Bounds::new(o, size(cw + px(0.5), b.size.height)), swatch(rgb_of(i as f32 / n as f32, 1., 1.))));
            }
            let m = point(b.origin.x + b.size.width * h - px(2.), b.origin.y);
            window.paint_quad(fill(Bounds::new(m, size(px(4.), b.size.height)), gpui::transparent_black()).border_widths(px(2.)).border_color(rgb(0xffffff)));
        };
        let drag = |f: fn(&mut Studio, Point<Pixels>, &mut Context<Studio>)| {
            move |this: &mut Studio, ev: &MouseMoveEvent, _: &mut gpui::Window, cx: &mut Context<Studio>| {
                if ev.pressed_button == Some(MouseButton::Left) {
                    f(this, ev.position, cx);
                }
            }
        };
        div()
            .pt(px(4.))
            .flex()
            .flex_col()
            .gap(px(8.))
            .child(
                div()
                    .id("color-sv")
                    .w(px(200.))
                    .h(px(96.))
                    .rounded(px(6.))
                    .overflow_hidden()
                    .cursor_crosshair()
                    .child(canvas(move |b, _, _| sv.set(Some(b)), move |b, _, window, _| paint_sv(b, window)).size_full())
                    .on_mouse_down(MouseButton::Left, cx.listener(|this, ev: &MouseDownEvent, _, cx| on_sv(this, ev.position, cx)))
                    .on_mouse_move(cx.listener(drag(on_sv))),
            )
            .child(
                div()
                    .id("color-hue")
                    .w(px(200.))
                    .h(px(12.))
                    .rounded(px(4.))
                    .overflow_hidden()
                    .cursor_pointer()
                    .child(canvas(move |b, _, _| hue.set(Some(b)), move |b, _, window, _| paint_hue(b, window)).size_full())
                    .on_mouse_down(MouseButton::Left, cx.listener(|this, ev: &MouseDownEvent, _, cx| on_hue(this, ev.position, cx)))
                    .on_mouse_move(cx.listener(drag(on_hue))),
            )
            .child(div().w(px(110.)).child(self.animate_hex.clone()))
            .into_any_element()
    }

    /// One line of a layer's options: a name, its value and - / + around it.
    fn layer_step(&self, i: usize, id: &'static str, name: &str, value: String, change: fn(&mut Layer, i32), cx: &mut Context<Self>) -> Div {
        div()
            .flex()
            .items_center()
            .gap(px(6.))
            .child(div().flex_1().text_size(px(12.)).text_color(rgb(TEXT())).child(name.to_string()))
            .child(icon_btn((id, 0usize), "minus").on_click(cx.listener(move |this, _, _, cx| this.edit_layer(i, |l| change(l, -1), cx))))
            .child(div().w(px(48.)).flex().justify_center().font_family(MONO).text_size(px(12.)).text_color(rgb(ACCENT())).child(value))
            .child(icon_btn((id, 1usize), "plus").on_click(cx.listener(move |this, _, _, cx| this.edit_layer(i, |l| change(l, 1), cx))))
    }

    /// The options of layer `i`: where it acts, its colour, how strong, how often and how many.
    fn layer_options(&mut self, i: usize, cx: &mut Context<Self>) -> AnyElement {
        let l = self.animate.settings.layers[i].clone();
        let editor = (matches!(l.area, Area::Color(_) | Area::Mask(_)) || l.effect.has_origin()).then(|| self.layer_editor(i, cx)).flatten();
        let color_open = self.animate.color_open;
        let picker = color_open.then(|| self.color_picker(cx));
        let row = || div().flex().flex_wrap().items_center().gap(px(4.));
        let tag = |name: &str| div().w(px(38.)).flex_none().text_size(px(11.5)).text_color(rgb(MUTED())).child(name.to_string());
        let mut colors: Vec<[u8; 3]> = ELEMENTS.iter().map(|e| e.color).collect();
        colors.extend(EXTRA_COLORS);
        let custom = l.color.filter(|c| !colors.contains(c));
        let area = area_index(&l.area);
        let mut body = div()
            .flex()
            .flex_col()
            .gap(px(10.))
            .p(px(10.))
            .rounded_b(px(7.))
            .bg(rgb(INPUT_BG()))
            .border_1()
            .border_color(rgb(BORDER_SOFT()))
            .child(row().child(tag(tr!("Zona", "Area"))).children((0..5).map(|k| {
                chip(("area", k), area_label(k), area == k).on_click(cx.listener(move |this, _, _, cx| {
                    // a painted or picked area starts from what the layer covered, to refine instead of redoing
                    let covered = this.layer_tint(i);
                    let first = this.animate_image().zip(covered.clone()).and_then(|(im, m)| {
                        let at = m.iter().position(|&b| b)? as u32;
                        let p = im.get_pixel(at % im.width(), at / im.width()).0;
                        Some([p[0], p[1], p[2]])
                    });
                    let next = match k {
                        0 => Area::Auto,
                        1 => Area::All,
                        2 => Area::Glow,
                        3 => Area::Color(first.unwrap_or([255, 255, 255])),
                        _ => Area::Mask(covered.map(|m| m.to_vec()).unwrap_or_default()),
                    };
                    this.animate.tool = if k == 3 { Tool::Pick } else { Tool::Paint };
                    this.edit_layer(
                        i,
                        |l| {
                            if area_index(&l.area) != k {
                                l.area = next;
                            }
                        },
                        cx,
                    );
                }))
            })))
            .children(editor)
            .child(
                row()
                    .child(tag("Color"))
                    .child(
                        chip("color-auto", "Auto", l.color.is_none()).on_click(cx.listener(move |this, _, _, cx| this.edit_layer(i, |l| l.color = None, cx))),
                    )
                    .children(colors.clone().into_iter().enumerate().map(|(k, c)| {
                        div()
                            .id(("color", k))
                            .w(px(20.))
                            .h(px(20.))
                            .rounded(px(6.))
                            .bg(rgb(hex(c)))
                            .border_2()
                            .border_color(rgb(if l.color == Some(c) { ACCENT() } else { BORDER() }))
                            .cursor_pointer()
                            .on_click(cx.listener(move |this, _, _, cx| this.edit_layer(i, |l| l.color = Some(c), cx)))
                    }))
                    .child(
                        div()
                            .id("color-custom")
                            .w(px(20.))
                            .h(px(20.))
                            .rounded(px(6.))
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(rgb(custom.map_or(RAISED(), hex)))
                            .border_2()
                            .border_color(rgb(if custom.is_some() || color_open { ACCENT() } else { BORDER() }))
                            .cursor_pointer()
                            .tooltip(tooltip(tr!("Otro color", "Another colour")))
                            .when(custom.is_none(), |d| d.child(icon("palette").size(px(12.)).text_color(rgb(MUTED()))))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.animate.color_open = !this.animate.color_open;
                                cx.notify();
                            })),
                    ),
            )
            .children(picker)
            .child(self.layer_step(
                i,
                "strength",
                tr!("Intensidad", "Strength"),
                format!("{:.0} %", l.strength * 100.),
                |l, d| l.strength = (l.strength + d as f32 * 0.25).clamp(0.25, 1.),
                cx,
            ))
            .child(self.layer_step(
                i,
                "repeats",
                tr!("Veces por ciclo", "Times per loop"),
                format!("×{}", l.repeats),
                |l, d| l.repeats = (l.repeats as i32 + d).clamp(1, 4) as usize,
                cx,
            ));
        if l.effect.amount().is_some() {
            body = body.child(self.layer_step(
                i,
                "amount",
                amount_label(l.effect),
                l.amount.to_string(),
                |l, d| {
                    if let Some((lo, hi, _)) = l.effect.amount() {
                        l.amount = (l.amount as i32 + d).clamp(lo as i32, hi as i32) as usize;
                    }
                },
                cx,
            ));
        }
        body.into_any_element()
    }

    /// The effects to add, grouped by what they do; picking one adds its layer and opens it.
    fn add_effects(&mut self, cx: &mut Context<Self>) -> AnyElement {
        div()
            .flex()
            .flex_col()
            .gap(px(8.))
            .p(px(10.))
            .mb(px(6.))
            .rounded(px(8.))
            .bg(rgb(INPUT_BG()))
            .border_1()
            .border_color(rgb(BORDER_SOFT()))
            .children(GROUPS.into_iter().enumerate().map(|(gi, group)| {
                div().flex().flex_col().gap(px(5.)).child(label(group_label(gi))).child(div().flex().flex_wrap().gap(px(5.)).children(group.iter().map(|&e| {
                    div()
                        .id(("add", e as usize))
                        .h(px(28.))
                        .pl(px(8.))
                        .pr(px(10.))
                        .flex()
                        .items_center()
                        .gap(px(6.))
                        .rounded(px(6.))
                        .bg(rgb(RAISED()))
                        .border_1()
                        .border_color(rgb(BORDER()))
                        .text_size(px(12.))
                        .text_color(rgb(TEXT()))
                        .cursor_pointer()
                        .hover(|s| s.bg(rgb(HOVER())))
                        .child(icon(effect_icon(e)).size(px(13.)).text_color(rgb(MUTED())))
                        .child(effect_label(e))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.animate.settings.layers.push(Layer::new(e));
                            this.animate.open = Some(this.animate.settings.layers.len() - 1);
                            this.animate.adding = false;
                            this.animate.color_open = false;
                            this.animate.element = None;
                            this.animate_changed(cx);
                        }))
                })))
            }))
            .into_any_element()
    }

    pub(super) fn animate_options(&mut self, cx: &mut Context<Self>) -> AnyElement {
        if self.animate_source().is_some_and(|(_, w, h)| w.max(h) > MAX_SIDE) {
            return div()
                .pt(px(12.))
                .text_size(px(12.))
                .text_color(rgb(WARN()))
                .child(tr!(
                    format!("La animación es para sprites de hasta {MAX_SIDE} px. Conviértelo antes en la pestaña Imagen."),
                    format!("Animation is for sprites up to {MAX_SIDE} px. Convert it first in the Image tab.")
                ))
                .into_any_element();
        }
        let a = &self.animate;
        let (element, frames, fps) = (a.element, a.settings.frames, a.fps);
        let layers: Vec<Layer> = a.settings.layers.clone();
        let open = a.open.filter(|&i| i < layers.len());
        let adding = a.adding || layers.is_empty();
        let mut open_body = open.map(|i| self.layer_options(i, cx));
        let add_list = adding.then(|| self.add_effects(cx));
        let mut list = div().flex().flex_col().gap(px(4.));
        for (i, l) in layers.iter().enumerate() {
            let is_open = open == Some(i);
            let (e, on, summary) = (l.effect, l.on, layer_summary(l));
            list = list.child(
                div()
                    .id(("layer", i))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .h(px(34.))
                    .pl(px(8.))
                    .pr(px(4.))
                    .when(is_open, |d| d.rounded_t(px(7.)))
                    .when(!is_open, |d| d.rounded(px(7.)))
                    .bg(rgb(if is_open { RAISED() } else { INPUT_BG() }))
                    .border_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(HOVER())))
                    .child(icon(effect_icon(e)).size(px(15.)).text_color(rgb(match l.color {
                        Some(c) if on => hex(c),
                        _ if on => TEXT(),
                        _ => MUTED_2(),
                    })))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .items_baseline()
                            .gap(px(8.))
                            .child(
                                div()
                                    .text_size(px(12.5))
                                    .font_weight(if is_open { FontWeight::SEMIBOLD } else { FontWeight::NORMAL })
                                    .text_color(rgb(if on { TEXT() } else { MUTED_2() }))
                                    .child(effect_label(e)),
                            )
                            .child(div().min_w_0().overflow_hidden().whitespace_nowrap().text_size(px(11.)).text_color(rgb(MUTED_2())).child(summary)),
                    )
                    .child(
                        div()
                            .id(("layer-on", i))
                            .w(px(26.))
                            .h(px(26.))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded(px(6.))
                            .hover(|s| s.bg(rgb(RAISED())))
                            .tooltip(tooltip(if on { tr!("Ocultar", "Hide") } else { tr!("Mostrar", "Show") }))
                            .child(icon(if on { "eye" } else { "eye-off" }).size(px(14.)).text_color(rgb(MUTED())))
                            .on_click(cx.listener(move |this, _, _, cx| {
                                cx.stop_propagation();
                                this.edit_layer(i, |l| l.on = !l.on, cx);
                            })),
                    )
                    .child(
                        div()
                            .id(("layer-remove", i))
                            .w(px(26.))
                            .h(px(26.))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded(px(6.))
                            .hover(|s| s.bg(rgb(RAISED())))
                            .tooltip(tooltip(tr!("Quitar", "Remove")))
                            .child(icon("trash-2").size(px(14.)).text_color(rgb(MUTED())))
                            .on_click(cx.listener(move |this, _, _, cx| {
                                cx.stop_propagation();
                                let layers = &mut this.animate.settings.layers;
                                if i < layers.len() {
                                    layers.remove(i);
                                }
                                this.animate.open = match this.animate.open {
                                    Some(o) if o == i => None,
                                    Some(o) if o > i => Some(o - 1),
                                    o => o,
                                };
                                this.animate.element = None;
                                this.animate_changed(cx);
                            })),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.animate.open = if this.animate.open == Some(i) { None } else { Some(i) };
                        this.animate.color_open = false;
                        cx.notify();
                    })),
            );
            if is_open {
                list = list.child(div().mt(px(-5.)).children(open_body.take()));
            }
        }
        let line = |name: &str| div().w(px(58.)).flex_none().text_size(px(12.)).text_color(rgb(MUTED())).child(name.to_string());
        div()
            .flex()
            .flex_col()
            .child(section(tr!("Elemento", "Element")))
            .child(div().flex().gap(px(6.)).children((0..ELEMENTS.len()).map(|i| {
                let active = element == Some(i);
                div()
                    .id(("element", i))
                    .w(px(38.))
                    .h(px(34.))
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded(px(7.))
                    .bg(rgb(RAISED()))
                    .border_2()
                    .border_color(rgb(if active { ACCENT() } else { BORDER() }))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(HOVER())))
                    .tooltip(tooltip(element_label(i)))
                    .child(icon(ELEMENT_ICONS[i]).size(px(17.)).text_color(rgb(hex(ELEMENTS[i].color))))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.animate.element = (this.animate.element != Some(i)).then_some(i);
                        if this.animate.element.is_some() {
                            this.animate.settings.layers = ELEMENTS[i].layers();
                            this.animate.open = None;
                            this.animate.adding = false;
                        }
                        this.animate_changed(cx);
                    }))
            })))
            .child(
                section(tr!("Efectos", "Effects")).child(
                    div()
                        .id("add-effect")
                        .h(px(24.))
                        .pl(px(6.))
                        .pr(px(9.))
                        .flex()
                        .items_center()
                        .gap(px(4.))
                        .rounded(px(6.))
                        .text_size(px(11.5))
                        .cursor_pointer()
                        .when(adding, |d| d.bg(rgb(ACCENT())).text_color(rgb(ACCENT_INK())))
                        .when(!adding, |d| d.bg(rgb(RAISED())).text_color(rgb(TEXT())).border_1().border_color(rgb(BORDER())).hover(|s| s.bg(rgb(HOVER()))))
                        .child(icon("plus").size(px(13.)).text_color(rgb(if adding { ACCENT_INK() } else { MUTED() })))
                        .child(tr!("Añadir", "Add"))
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.animate.adding = !this.animate.adding;
                            cx.notify();
                        })),
                ),
            )
            .children(add_list)
            .child(list)
            .child(
                section(tr!("Bucle", "Loop"))
                    .child(div().font_family(MONO).text_size(px(10.5)).text_color(rgb(MUTED_2())).child(format!("{:.1} s", frames as f32 / fps as f32))),
            )
            .child(div().flex().items_center().gap(px(6.)).pb(px(6.)).child(line("Frames")).child(div().flex().flex_wrap().gap(px(5.)).children(
                FRAMES.into_iter().map(|n| {
                    chip(("frames", n), &n.to_string(), frames == n).on_click(cx.listener(move |this, _, _, cx| {
                        this.animate.settings.frames = n;
                        this.animate_changed(cx);
                    }))
                }),
            )))
            .child(div().flex().items_center().gap(px(6.)).child(line("ms / frame")).child(div().flex().flex_wrap().gap(px(5.)).children(FPS.into_iter().map(
                |v| {
                    chip(("afps", v as usize), &(1000 / v).to_string(), fps == v).on_click(cx.listener(move |this, _, _, cx| {
                        this.animate.fps = v;
                        this.animate_changed(cx);
                    }))
                },
            ))))
            .into_any_element()
    }

    /// The Advanced section of the mode: how much of the item counts as glowing, and another placement.
    pub(super) fn animate_advanced(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let s = &self.animate.settings;
        let rows: [AdvancedRow; 2] = [
            (
                tr!("Zona que brilla", "Glowing area"),
                tr!("Más = más partes del objeto cuentan como brillo", "More = more of the item counts as glow"),
                format!("{:.2}", 1. - s.threshold),
                |s, d| s.threshold = (s.threshold - d as f32 * 0.05).clamp(0.2, 0.9),
            ),
            (
                tr!("Variación", "Variation"),
                tr!("Otra colocación de chispas, rayos y llamas", "Another placement of sparks, arcs and flames"),
                s.seed.to_string(),
                |s, d| s.seed = (s.seed as i64 + d as i64).max(0) as u64,
            ),
        ];
        div()
            .flex()
            .flex_col()
            .children(rows.into_iter().enumerate().map(|(i, (name, hint, value, change))| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .py(px(4.))
                    .child(
                        div()
                            .id(("anim-adv", i))
                            .flex_1()
                            .text_size(px(12.5))
                            .text_color(rgb(TEXT()))
                            .when(!hint.is_empty(), |d| d.tooltip(tooltip(hint)))
                            .child(name.to_string()),
                    )
                    .child(icon_btn(("anim-adv-minus", i), "minus").on_click(cx.listener(move |this, _, _, cx| {
                        change(&mut this.animate.settings, -1);
                        this.animate_changed(cx);
                    })))
                    .child(div().w(px(52.)).flex().justify_center().font_family(MONO).text_size(px(12.5)).text_color(rgb(ACCENT())).child(value))
                    .child(icon_btn(("anim-adv-plus", i), "plus").on_click(cx.listener(move |this, _, _, cx| {
                        change(&mut this.animate.settings, 1);
                        this.animate_changed(cx);
                    })))
            }))
            .into_any_element()
    }
}

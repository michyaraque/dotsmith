//! The Sprite Studio window: mode rail | controls | live stage | history.
use gpui::{
    AnyElement, Bounds, ClickEvent, ClipboardEntry, Context, Div, Entity, ExternalPaths, FocusHandle, FontWeight, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, ObjectFit, PathPromptOptions, Pixels, Point, RenderImage, SharedString, Stateful, Window, WindowControlArea, actions, canvas, div, fill, img,
    point, prelude::*, px, rgb, rgba, size,
};
use image::RgbaImage;
use serde_json::{Value, json};
use std::cell::Cell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::backend::{Backend, Folders, Live, Msg};
use crate::settings::{Settings, WindowState};
use crate::tr;

#[path = "creature_mode.rs"]
mod creature_mode;
#[path = "model_hub.rs"]
mod model_hub;
#[path = "model_picker.rs"]
mod model_picker;
#[path = "palette.rs"]
mod palette;
#[path = "settings_view.rs"]
mod settings_view;
#[path = "tools_menu.rs"]
mod tools_menu;
#[path = "welcome.rs"]
mod welcome;
use crate::icons::icon;
use crate::input::TextInput;
use crate::pixels::{self, Item};
use crate::rig;
use crate::theme::*;
use model_picker::{FamilyInfo, PickerState};

/// Height of the result details card under the viewer (fixed, so the viewer square is computed, not measured).
const DETAILS_H: f32 = 132.;

actions!(
    studio,
    [
        Generate,
        Stop,
        PasteImage,
        CopyImage,
        ClosePromptModal,
        DeleteSelected,
        UndoDelete,
        PrevResult,
        NextResult,
        OpenSettings,
        OpenModelPicker,
        OpenPalette,
        FocusNext,
        FocusPrev
    ]
);

/// How long a deleted result waits (undoable) before it goes to the Recycle Bin.
const UNDO_SECS: u64 = 15;
/// The last deleted result while it can be undone: (original, parked) file pairs, its history position, when.
type Parked = (Vec<(PathBuf, PathBuf)>, usize, Instant);

/// Set while the window's background is built under an open modal: its controls are then no tab stops, so Tab stays
/// inside the modal.
static TAB_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

thread_local! {
    /// Focus handles of the keyboard-reachable controls, one per element id for the life of the window, plus spare ones
    /// made at the start of each frame (a handle needs the app context; the control helpers don't have it). gpui's own
    /// per-element handles (`tab_index`) focus and click but never get their focus style, explicit ones do.
    static KB: std::cell::RefCell<(Vec<FocusHandle>, HashMap<String, FocusHandle>)> = Default::default();
}

fn kb_refill(cx: &mut gpui::App) {
    KB.with(|k| {
        let mut k = k.borrow_mut();
        while k.0.len() < 512 {
            k.0.push(cx.focus_handle().tab_stop(true));
        }
    });
}

/// A clickable element reachable with Tab: Enter / Space click it (gpui), a ring shows where the focus is.
pub(crate) trait Keyboard {
    fn kb(self, id: impl Into<gpui::ElementId>) -> Stateful<Div>;
}

impl Keyboard for Div {
    fn kb(self, id: impl Into<gpui::ElementId>) -> Stateful<Div> {
        let id: gpui::ElementId = id.into();
        let key = id.to_string();
        let d = self.id(id);
        if TAB_OFF.load(std::sync::atomic::Ordering::Relaxed) {
            return d;
        }
        let handle = KB.with(|k| {
            let mut k = k.borrow_mut();
            if let Some(h) = k.1.get(&key) {
                return Some(h.clone());
            }
            let h = k.0.pop()?;
            k.1.insert(key, h.clone());
            Some(h)
        });
        match handle {
            Some(h) => d.track_focus(&h).focus(|s| s.border_1().border_color(rgb(ACCENT_DIM()))),
            None => d,
        }
    }
}

pub fn bind_keys(cx: &mut gpui::App) {
    use gpui::KeyBinding;
    cx.bind_keys([
        KeyBinding::new("ctrl-enter", Generate, None),
        KeyBinding::new("escape", Stop, None),
        KeyBinding::new("escape", ClosePromptModal, Some("PromptModal")),
        KeyBinding::new("ctrl-v", PasteImage, Some("Studio")),
        KeyBinding::new("ctrl-c", CopyImage, Some("Studio")),
        KeyBinding::new("delete", DeleteSelected, Some("Studio")),
        KeyBinding::new("ctrl-z", UndoDelete, Some("Studio")),
        KeyBinding::new("tab", FocusNext, None),
        KeyBinding::new("shift-tab", FocusPrev, None),
        KeyBinding::new("left", PrevResult, Some("Studio")),
        KeyBinding::new("right", NextResult, Some("Studio")),
        KeyBinding::new("ctrl-,", OpenSettings, None),
        KeyBinding::new("ctrl-l", OpenModelPicker, None),
        KeyBinding::new("ctrl-k", OpenPalette, None),
        KeyBinding::new("ctrl-p", OpenPalette, None),
    ]);
    palette::bind_keys(cx);
}

/// Lines of the prompt field: (min, max) in the side panel, and in the enlarged editor.
const PANEL_LINES: (usize, usize) = (4, 9);
const MODAL_LINES: (usize, usize) = (16, 26);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Single,
    Pack,
    Variants,
    Regions,
    Photo,
    Angle,
    Pose,
    Creature,
    Walls,
}
/// Rail order: what is made from a description, then what starts from an image.
const MODES: [Mode; 9] = [Mode::Single, Mode::Pack, Mode::Walls, Mode::Creature, Mode::Variants, Mode::Regions, Mode::Angle, Mode::Photo, Mode::Pose];
const CREATE_MODES: usize = 4;

impl Mode {
    fn key(self) -> &'static str {
        match self {
            Mode::Single => "single",
            Mode::Pack => "pack",
            Mode::Variants => "variants",
            Mode::Regions => "regions",
            Mode::Photo => "photo",
            Mode::Angle => "angle",
            Mode::Pose => "pose",
            Mode::Creature => "creature",
            Mode::Walls => "walls",
        }
    }
    fn from_key(k: &str) -> Option<Mode> {
        MODES.into_iter().find(|m| m.key() == k)
    }
    fn label(self) -> &'static str {
        match self {
            Mode::Single => "Sprite",
            Mode::Pack => "Pack",
            Mode::Variants => tr!("Variantes", "Variants"),
            Mode::Regions => tr!("Zonas", "Regions"),
            Mode::Photo => tr!("Imagen", "Image"),
            Mode::Angle => tr!("Ángulo", "Angle"),
            Mode::Pose => "Pose",
            Mode::Creature => tr!("Personaje", "Character"),
            Mode::Walls => tr!("Muros", "Walls"),
        }
    }
    fn glyph(self) -> &'static str {
        match self {
            Mode::Single => "type",
            Mode::Pack => "grid-3x3",
            Mode::Variants => "layers",
            Mode::Regions => "brush",
            Mode::Photo => "image",
            Mode::Angle => "rotate-3d",
            Mode::Pose => "person-standing",
            Mode::Creature => "footprints",
            Mode::Walls => "brick-wall",
        }
    }
    fn title(self) -> &'static str {
        match self {
            Mode::Single => tr!("Sprite desde texto", "Sprite from text"),
            Mode::Pack => tr!("Pack de objetos", "Item pack"),
            Mode::Variants => tr!("Variantes de un sprite", "Sprite variants"),
            Mode::Regions => tr!("Edición por zonas", "Region edit"),
            Mode::Photo => tr!("Imagen a sprite", "Image to sprite"),
            Mode::Angle => tr!("Cambio de ángulo", "Change angle"),
            Mode::Pose => tr!("Pose y silueta", "Pose and silhouette"),
            Mode::Creature => tr!("Personaje animado", "Animated character"),
            Mode::Walls => tr!("Set de muros", "Wall set"),
        }
    }
    fn uses_source(self) -> bool {
        matches!(self, Mode::Variants | Mode::Regions | Mode::Photo | Mode::Angle)
    }
}

/// 16 px only for models that declare it; the others start at 32.
const SINGLE_SIZES: [(&str, u32, u32); 9] = [
    ("16", 16, 16),
    ("32", 32, 32),
    ("48", 48, 48),
    ("64", 64, 64),
    ("96", 96, 96),
    ("128", 128, 128),
    ("192", 192, 192),
    ("64×128", 64, 128),
    ("96×144", 96, 144),
];
const PACKS: [(&str, u32, u32); 5] = [("3×3 · 32", 3, 32), ("2×2 · 48", 2, 48), ("2×2 · 64", 2, 64), ("3×3 · 48", 3, 48), ("3×3 · 64", 3, 64)];
const OUT_SIZES: [u32; 3] = [32, 64, 96];
const DIRECT_SIZES: [u32; 5] = [32, 48, 64, 96, 128];
const CAMERAS: [(&str, &str); 3] = [("Normal", "Normal"), ("45°", "Cenital 45° (propia)"), ("45° recto", "Cenital 45° · objeto recto")];

fn camera_label(i: usize) -> &'static str {
    match i {
        0 => tr!("Normal", "Normal"),
        1 => tr!("Cenital 45°", "Top-down 45°"),
        _ => tr!("45° recto", "45° straight"),
    }
}

struct Form {
    prompt: Entity<TextInput>,
    seed: Entity<TextInput>,
    camera: usize,
    size: usize,
    opaque: bool,
    center: bool,
    mirror: bool,
    strength: f32,
    variants: u32,
    guidance: f32,
    steps: Option<u32>, // None: the loaded base's default
    turbo_steps: Option<u32>,
    pitch: f32,
    yaw: f32,
    warp: bool,
    all_dirs: bool, // Personaje: the 4 directions or only `dir`
    dir: usize,
    kind: usize,   // Personaje with a LoRA walk format: which of its kinds
    room: bool,    // Muros: also assemble the 3x3 closed room
    restyle: bool, // Imagen entera: then redraw with the LoRA (img2sprite)
}

struct Source {
    path: PathBuf,
    view: Arc<RenderImage>,
    w: u32,
    h: u32,
    sprite: Option<RgbaImage>, // the 32x32 the region mask paints on (from the backend, exactly as it samples)
}

#[derive(Clone)]
struct Run {
    scale: u32,
    ckpts: Vec<(String, String)>,
    run: String,
    family: String,
    name: String,
    version: String,
    age: String,
    magenta: bool,
}

#[derive(Clone, Default)]
struct GpuState {
    ok: bool,
    name: String,
    vram_mb: u64,
    error: String,
    low_vram: bool,
    memory: Vec<dotsmith_engine::core::memory::Placed>,
}

#[derive(Clone)]
struct FileRow {
    role: String,
    turbo_only: bool,
    path: String,
    found: bool,
    dest: Option<PathBuf>,
    remote: Option<RemoteFile>,
    /// an optional alternative in another precision (e.g. "bf16"); never required
    precision: Option<String>,
}

#[derive(Clone)]
struct RemoteFile {
    repo: String,
    path: String,
    url: String,
}

#[derive(Default)]
struct ModelState {
    families: Vec<FamilyInfo>,
    recent: Vec<(String, String, String, String)>,
    loaded: Option<String>,
    base: String,
    family: String,
    lora: String,
    run: String,
    version: String,
    scale: u32,
    out_dir: PathBuf,
    runs: Vec<Run>,
    turbo: bool,
    /// the current LoRA's own walk-sheet format (catalog::WalkFormat), if it declares one
    walk: Option<dotsmith_engine::catalog::WalkFormat>,
    gpu: Option<GpuState>,
    files: Vec<(String, Vec<FileRow>)>,
    catalog_error: Option<String>,
    /// (repo, label) of models listed as coming soon
    store: Vec<(String, String)>,
}

impl ModelState {
    fn files_of(&self, family: &str) -> &[FileRow] {
        self.files.iter().find(|f| f.0 == family).map(|f| f.1.as_slice()).unwrap_or(&[])
    }

    fn family_ready(&self, family: &str) -> bool {
        let f = self.files_of(family);
        // every required file is on disk in at least one precision (default or an alternative such as bf16)
        let covered = |r: &FileRow| r.found || f.iter().any(|o| o.found && o.role == r.role && o.turbo_only == r.turbo_only);
        !f.is_empty() && f.iter().all(|r| r.turbo_only || r.precision.is_some() || covered(r))
    }

    fn can_generate(&self) -> bool {
        self.gpu.as_ref().is_some_and(|g| g.ok) && self.family_ready(&self.family)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Tool {
    Brush,
    Erase,
    Wand,
}

#[derive(Clone, Copy, PartialEq)]
enum Level {
    Info,
    Ok,
    Error,
}

pub struct Studio {
    backend: Option<Arc<Backend>>,
    focus: FocusHandle,
    mode: Mode,
    forms: Vec<Form>,
    model: ModelState,
    activity: Live, // model loading / queued switch, read from the worker every tick
    phase: String,
    source: Option<Source>,
    mask: Vec<bool>,
    tool: Tool,
    brush: i32,
    painting: Option<bool>,
    grid: Rc<Cell<Option<Bounds<Pixels>>>>,
    gizmo_drag: Option<(Point<Pixels>, f32, f32)>,
    pending: bool,
    running: bool,
    started: Option<Instant>,
    progress: (u32, u32, u32, u32), // step, total, index, count
    aurora: crate::aurora::Aurora,  // the live preview
    job_results: Vec<PathBuf>,
    history: Vec<Item>,
    only_mode: bool,
    selected: Option<PathBuf>,
    big: Option<Arc<RgbaImage>>,
    status: Option<(String, Level, Instant)>,
    model_open: bool,
    prompt_modal: bool,
    small: Option<(Arc<RenderImage>, Arc<RenderImage>, u32, u32)>, // sprite at 1x and 2x real pixels
    zoom: f32,
    pan: Point<Pixels>,
    view_drag: Option<(Point<Pixels>, Point<Pixels>)>,
    view_area: Rc<Cell<Option<Bounds<Pixels>>>>,
    scale_factor: f32,
    maximized: bool,
    viewport: (f32, f32),
    restore_size: (f32, f32),
    square: Rc<Cell<Option<Bounds<Pixels>>>>,
    logo: Arc<RenderImage>,
    picker: PickerState,
    anim: crate::anim::Anim,
    trash: Vec<Arc<RenderImage>>,
    last_tick: Instant,
    sys: Option<crate::sysmon::Sample>,
    _quit: Option<gpui::Subscription>,
    pose: rig::Pose,
    pose_sheet: bool,
    pose_playing: bool,
    pose_tick: Instant,
    menu_open: bool,
    settings: Settings,
    settings_open: bool,
    settings_tab: settings_view::Tab,
    downloads: crate::download::Downloader,
    hf: crate::hf::Hub,
    hub_open: bool,
    hub_sel: Option<String>,
    hub_query: Entity<TextInput>,
    downloads_open: bool,
    hf_token: Entity<TextInput>,
    hf_url: Entity<TextInput>,
    advanced_open: bool,
    lora_scale: f32,
    /// --autogen: generate once the model is ready (end-to-end checks of the live view)
    autogen: bool,
    palette_open: bool,
    palette_query: Entity<TextInput>,
    palette_sel: usize,
    palette_last: String,
    palette_scroll: gpui::ScrollHandle,
    /// Some(true) while the start screen is up after a launch that could not generate yet
    onboarding: Option<bool>,
    /// child index in the list of each visible command (group headers sit between them)
    palette_rows: Vec<usize>,
    /// right-click menu of a history result, at this window position
    ctx_menu: Option<Point<Pixels>>,
    deleted: Option<Parked>,
    hist_query: Entity<TextInput>,
    /// the history search text, read once per frame
    hist_query_text: String,
}

fn folders(s: &Settings) -> Folders {
    Folders { library: s.lora_library(), output: s.output_dir.clone(), fallback: Settings::fallback_output_dir() }
}

fn camera_index(name: &str) -> usize {
    CAMERAS.iter().position(|c| c.1 == name).unwrap_or(0)
}

impl Studio {
    pub fn new(settings: Settings, window: &mut Window, cx: &mut Context<Self>) -> Self {
        // stored in enum order: forms[mode as usize]; MODES is only the rail order
        let forms: Vec<Form> = [Mode::Single, Mode::Pack, Mode::Variants, Mode::Regions, Mode::Photo, Mode::Angle, Mode::Pose, Mode::Creature, Mode::Walls]
            .iter()
            .map(|m| {
                let ph = match m {
                    Mode::Single => "A short silver wand with a bright blue crystal, a dark leather grip and gold rings.",
                    Mode::Pack => "Nine elemental wands. Row 1: fire, ice, nature. Row 2: lightning, holy, shadow. Row 3: water, poison, arcane.",
                    Mode::Variants => "A crimson leather cover with an orange flame emblem and a red bookmark ribbon.",
                    Mode::Regions => "Turn the selected gem into a glowing green emerald.",
                    Mode::Photo => "An orc warrior in iron armour holding a spear.",
                    Mode::Angle => "A wooden treasure chest with iron bands and a gold lock.",
                    Mode::Pose => "",
                    Mode::Creature => "Skeleton warrior, bone white body, rusty iron helmet, round wooden shield, undead.",
                    Mode::Walls => "cotton candy, fluffy pastel pink spun sugar, baby blue swirls, white sugar sparkles, puffy rounded top, candy land wall",
                };
                let lines = if *m == Mode::Pack { 6 } else { 4 };
                Form {
                    prompt: cx.new(|cx| TextInput::new(cx, ph, lines).pad_right(28.).with_max_lines(if *m == Mode::Pack { 11 } else { PANEL_LINES.1 })),
                    seed: cx.new(|cx| TextInput::new(cx, tr!("aleatoria", "random"), 1).mono()),
                    camera: 0,
                    size: match m {
                        Mode::Photo => 2,
                        Mode::Single => 1,
                        _ => 0,
                    },
                    opaque: false,
                    center: false,
                    mirror: false,
                    strength: match m {
                        Mode::Regions => 0.8,
                        Mode::Angle => 0.65,
                        _ => 0.45,
                    },
                    variants: match m {
                        Mode::Regions => 1,
                        _ => 3,
                    },
                    guidance: match m {
                        Mode::Regions => 5.0,
                        _ => 3.0,
                    },
                    steps: (*m == Mode::Creature).then_some(30),
                    turbo_steps: None,
                    pitch: 45.,
                    yaw: 0.,
                    warp: true,
                    all_dirs: true,
                    dir: 0,
                    kind: 0,
                    room: true,
                    restyle: true,
                }
            })
            .collect();
        if let Some(text) = std::env::args().skip_while(|a| a != "--prompt").nth(1) {
            let start_mode =
                std::env::args().skip_while(|a| a != "--mode").nth(1).or(settings.last_mode.clone()).and_then(|k| Mode::from_key(&k)).unwrap_or(Mode::Single);
            forms[start_mode as usize].prompt.update(cx, |t, cx| t.set_text(&text, cx));
        }
        let (backend, status) = match Backend::start(folders(&settings)) {
            Ok(b) => (Some(b), None),
            Err(e) => (None, Some((e, Level::Error, Instant::now()))),
        };
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(Duration::from_millis(30)).await;
                if this.update(cx, |s, cx| s.pump(cx)).is_err() {
                    break;
                }
            }
        })
        .detach();
        let quit = cx.on_app_quit(|this: &mut Studio, _| {
            this.finish_delete();
            if let Some(b) = this.backend.take() {
                b.shutdown();
            }
            this.settings.last_mode = Some(this.mode.key().to_string());
            let (width, height) = this.restore_size;
            this.settings.window = Some(WindowState { width, height, maximized: this.maximized });
            let _ = this.settings.save();
            async {}
        });
        let focus = cx.focus_handle();
        window.focus(&focus);
        let token = settings.hf_token.clone().unwrap_or_default();
        let hf_token = cx.new(|cx| {
            let mut t = TextInput::new(cx, "hf_…", 1).mono();
            t.set_text(&token, cx);
            t
        });
        let hf_url = cx.new(|cx| TextInput::new(cx, "owner/repo/path/file.safetensors", 1).mono());
        let hub_query = cx.new(|cx| TextInput::new(cx, tr!("Buscar por nombre o repo…", "Search by name or repo…"), 1));
        let palette_query = cx.new(|cx| {
            let mut t = TextInput::new(cx, tr!("Buscar modos, ajustes, herramientas…", "Search modes, settings, tools…"), 1).bare();
            if let Some(q) = std::env::args().skip_while(|a| a != "--open-palette").nth(1).filter(|q| !q.starts_with("--")) {
                t.set_text(&q, cx);
            }
            t
        });
        if std::env::args().any(|a| a == "--open-palette") {
            window.focus(&palette_query.read(cx).focus_handle.clone());
        }
        // --demo-live: a fake half-denoised preview, for screenshots of the live view without a model
        let demo = std::env::args().any(|a| a == "--demo-live");
        let mut aurora = crate::aurora::Aurora::default();
        if demo {
            let im = image::RgbaImage::from_fn(256, 256, |x, y| {
                let (fx, fy) = (x as f32 / 256., y as f32 / 256.);
                let blob = (-(((fx - 0.45).powi(2) + (fy - 0.5).powi(2)) * 9.)).exp();
                let c = |a: f32, b: f32, d: f32| ((a * (1. - fx) + b * fx) * (1. - blob) + d * blob).clamp(0., 255.) as u8;
                image::Rgba([c(110., 170., 210.), c(90. + 60. * fy, 190. - 80. * fy, 130.), c(220., 120. + 60. * fy, 70.), 255])
            });
            aurora.push(crate::aurora::colors(&im, None));
        }
        Studio {
            backend,
            focus,
            forms,
            mode: std::env::args().skip_while(|a| a != "--mode").nth(1).or(settings.last_mode.clone()).and_then(|k| Mode::from_key(&k)).unwrap_or(Mode::Single),
            model: ModelState::default(),
            activity: Live::default(),
            phase: "off".into(),
            source: None,
            mask: vec![false; 1024],
            tool: Tool::Brush,
            brush: 1,
            painting: None,
            grid: Rc::new(Cell::new(None)),
            gizmo_drag: None,
            pending: false,
            running: demo,
            started: demo.then(Instant::now),
            progress: if demo { (3, 8, 0, 1) } else { (0, 0, 0, 0) },
            aurora,
            job_results: vec![],
            history: vec![],
            only_mode: false,
            selected: None,
            big: None,
            status,
            model_open: std::env::args().any(|a| a == "--open-model"), // --open-model: isolated screenshots of the selector
            prompt_modal: false,
            small: None,
            zoom: 1.0,
            pan: Point::default(),
            view_drag: None,
            view_area: Rc::new(Cell::new(None)),
            scale_factor: 1.0,
            maximized: false,
            viewport: (1500., 940.),
            square: Rc::new(Cell::new(None)),
            logo: pixels::render(&image::load_from_memory(include_bytes!("../assets/dotsmith.png")).unwrap().to_rgba8(), 1),
            picker: PickerState::default(),
            anim: crate::anim::Anim::default(),
            trash: vec![],
            last_tick: Instant::now(),
            sys: None,
            _quit: Some(quit),
            pose: rig::Pose::default(),
            pose_sheet: false,
            pose_playing: true,
            pose_tick: Instant::now(),
            menu_open: std::env::args().any(|a| a == "--open-tools"),
            settings_open: std::env::args().any(|a| a == "--open-settings"),
            restore_size: (1500., 940.),
            downloads: Default::default(),
            hf: Default::default(),
            hf_token,
            hf_url,
            hub_open: std::env::args().any(|a| a == "--open-hub"),
            hub_sel: std::env::args().skip_while(|a| a != "--open-hub").nth(1).filter(|a| !a.starts_with("--")),
            hub_query,
            downloads_open: std::env::args().any(|a| a == "--open-downloads"),
            advanced_open: std::env::args().any(|a| a == "--advanced"),
            lora_scale: settings.lora_scale.unwrap_or(1.0),
            autogen: std::env::args().any(|a| a == "--autogen"),
            palette_open: std::env::args().any(|a| a == "--open-palette"),
            palette_query,
            palette_sel: 0,
            palette_last: String::new(),
            palette_scroll: gpui::ScrollHandle::new(),
            onboarding: None,
            palette_rows: vec![],
            ctx_menu: None,
            deleted: None,
            hist_query: cx.new(|cx| TextInput::new(cx, tr!("Buscar en el historial…", "Search history…"), 1)),
            hist_query_text: String::new(),
            settings,
            settings_tab: std::env::args().skip_while(|a| a != "--open-settings").nth(1).and_then(|k| settings_view::Tab::from_key(&k)).unwrap_or_default(),
        }
    }

    /// Default steps of the selected base (Turbo or not).
    fn default_steps(&self) -> u32 {
        let t = self.model.turbo;
        self.family_info().map(|i| i.steps[t as usize]).unwrap_or(if t { 8 } else { 25 })
    }

    fn steps_of(&self, f: &Form) -> u32 {
        let d = self.default_steps();
        if self.model.turbo { f.turbo_steps.unwrap_or(d) } else { f.steps.unwrap_or(d) }
    }

    fn family_info(&self) -> Option<&FamilyInfo> {
        self.model.families.iter().find(|f| f.id == self.model.family)
    }

    /// Whether the loaded family can run `m` (Pose needs no model).
    fn mode_ok(&self, m: Mode) -> bool {
        m == Mode::Pose || self.family_info().is_none_or(|f| f.modes.is_empty() || f.modes.iter().any(|k| k == m.key()))
    }

    fn size_ok(&self, w: u32, h: u32) -> bool {
        self.family_info().is_none_or(|f| f.sizes.is_empty() || (w == h && f.sizes.contains(&w)))
    }

    fn plain_family(&self) -> bool {
        self.family_info().is_some_and(|f| f.plain)
    }

    fn form(&self) -> &Form {
        &self.forms[self.mode as usize]
    }
    fn form_mut(&mut self) -> &mut Form {
        &mut self.forms[self.mode as usize]
    }
    fn say(&mut self, msg: impl Into<String>, level: Level) {
        self.status = Some((msg.into(), level, Instant::now()));
    }
    fn send(&mut self, v: Value) {
        match &self.backend {
            Some(b) => b.send(v),
            None => self.say(tr!("El motor no está en marcha. Reinicia Dotsmith.", "The engine is not running. Restart Dotsmith."), Level::Error),
        }
    }

    // ------------------------------------------------------------ backend events
    fn pump(&mut self, cx: &mut Context<Self>) {
        let msgs = self.backend.as_ref().map(|b| b.poll()).unwrap_or_default();
        let any = !msgs.is_empty();
        for m in msgs {
            self.apply(m, cx);
        }
        if self.mode == Mode::Pose && self.pose_playing && self.pose_tick.elapsed() >= Duration::from_millis(125) {
            self.pose_tick = Instant::now();
            self.pose.frame = (self.pose.frame + 1) % rig::FRAMES;
            cx.notify();
        }
        // a selected creature sheet plays its walk cycle
        if let Some(n) = self.selected_item().and_then(|i| creature_mode::SheetInfo::of(&i.json)).map(|s| s.len())
            && self.anim.advance(n)
        {
            cx.notify();
        }
        // the worker's model activity is read directly (it can sit inside a 30 s load without sending events); a change,
        // or an ongoing load whose timer and bar must move, repaints
        if let Some(b) = self.backend.as_ref() {
            let a = b.live();
            let changed = a.loading.as_ref().map(|l| (&l.what, l.since)) != self.activity.loading.as_ref().map(|l| (&l.what, l.since))
                || a.queued.as_ref().map(|q| q.0) != self.activity.queued.as_ref().map(|q| q.0);
            self.activity = a;
            if changed {
                cx.notify();
            }
        }
        if self.hf.take_changed() {
            cx.notify();
        }
        let sys = crate::sysmon::latest();
        if sys != self.sys {
            self.sys = sys;
            cx.notify();
        }
        if let Some(done) = self.downloads.take_finished() {
            match done {
                Ok(n) => self.say(tr!(format!("Descarga completa · {n} archivos."), format!("Download complete · {n} files.")), Level::Ok),
                Err(e) => self.say(e, Level::Error),
            }
            if let Some(b) = &self.backend {
                b.configure(folders(&self.settings));
            }
            cx.notify();
        }
        if self.deleted.as_ref().is_some_and(|d| d.2.elapsed() > Duration::from_secs(UNDO_SECS)) {
            self.finish_delete();
            cx.notify();
        }
        let busy = self.running || self.activity.loading.is_some() || self.activity.queued.is_some() || self.downloads.progress().running;
        let tick = busy && self.last_tick.elapsed() > Duration::from_millis(250);
        if any || tick {
            self.last_tick = Instant::now();
            cx.notify();
        }
    }

    fn apply(&mut self, m: Msg, cx: &mut Context<Self>) {
        match m {
            Msg::Event(v) => match v["ev"].as_str().unwrap_or("") {
                "state" => {
                    let out = PathBuf::from(v["out_dir"].as_str().unwrap_or(""));
                    let reload = out != self.model.out_dir;
                    self.model = ModelState {
                        families: FamilyInfo::parse(&v["families"]),
                        recent: v["recent"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .map(|r| {
                                        let s = |k: &str| r[k].as_str().unwrap_or("").to_string();
                                        (s("base"), s("lora"), s("label"), s("family"))
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                        loaded: v["loaded"].as_str().map(String::from),
                        base: v["base"].as_str().unwrap_or("").into(),
                        family: v["family"].as_str().unwrap_or("").into(),
                        lora: v["lora"].as_str().unwrap_or("").into(),
                        run: v["run"].as_str().unwrap_or("").into(),
                        version: v["version"].as_str().unwrap_or("").into(),
                        scale: v["scale"].as_u64().unwrap_or(16) as u32,
                        walk: serde_json::from_value(v["walk"].clone()).ok(),
                        out_dir: out,
                        turbo: v["turbo"].as_bool().unwrap_or(false),
                        runs: v["runs"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .map(|r| Run {
                                        scale: r["scale"].as_u64().unwrap_or(16) as u32,
                                        run: r["run"].as_str().unwrap_or("").into(),
                                        family: r["family"].as_str().unwrap_or("").into(),
                                        name: r["name"].as_str().unwrap_or("").into(),
                                        version: r["version"].as_str().unwrap_or("").into(),
                                        age: r["age"].as_str().unwrap_or("").into(),
                                        magenta: r["magenta"].as_bool().unwrap_or(false),
                                        ckpts: r["ckpts"]
                                            .as_array()
                                            .map(|c| {
                                                c.iter()
                                                    .map(|k| (k["label"].as_str().unwrap_or("").to_string(), k["path"].as_str().unwrap_or("").to_string()))
                                                    .collect()
                                            })
                                            .unwrap_or_default(),
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                        gpu: v["gpu"].as_object().map(|_| GpuState {
                            ok: v["gpu"]["ok"].as_bool().unwrap_or(false),
                            name: v["gpu"]["name"].as_str().unwrap_or("").into(),
                            vram_mb: v["gpu"]["vram_mb"].as_u64().unwrap_or(0),
                            error: v["gpu"]["error"].as_str().unwrap_or("").into(),
                            low_vram: v["gpu"]["low_vram"].as_bool().unwrap_or(false),
                            memory: serde_json::from_value(v["gpu"]["memory"].clone()).unwrap_or_default(),
                        }),
                        files: v["files"]
                            .as_object()
                            .map(|m| {
                                m.iter()
                                    .map(|(fam, rows)| {
                                        (
                                            fam.clone(),
                                            rows.as_array()
                                                .map(|a| {
                                                    a.iter()
                                                        .map(|r| FileRow {
                                                            role: r["role"].as_str().unwrap_or("").into(),
                                                            turbo_only: r["variant"].as_str().is_some(),
                                                            precision: r["precision"].as_str().map(String::from),
                                                            path: r["path"].as_str().unwrap_or("").into(),
                                                            found: r["found"].as_bool().unwrap_or(false),
                                                            dest: r["dest"].as_str().map(PathBuf::from),
                                                            remote: r["remote"].as_object().map(|_| RemoteFile {
                                                                repo: r["remote"]["repo"].as_str().unwrap_or("").into(),
                                                                path: r["remote"]["path"].as_str().unwrap_or("").into(),
                                                                url: r["remote"]["url"].as_str().unwrap_or("").into(),
                                                            }),
                                                        })
                                                        .collect()
                                                })
                                                .unwrap_or_default(),
                                        )
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                        catalog_error: v["catalog_error"].as_str().map(String::from),
                        store: v["store"]
                            .as_array()
                            .map(|a| a.iter().map(|s| (s["repo"].as_str().unwrap_or("").to_string(), s["label"].as_str().unwrap_or("").to_string())).collect())
                            .unwrap_or_default(),
                    };
                    self.phase = v["phase"].as_str().unwrap_or("off").into();
                    if self.onboarding.is_none() && self.model.gpu.is_some() {
                        self.onboarding = Some(!self.model.can_generate() || std::env::args().any(|a| a == "--welcome"));
                    }
                    if self.autogen && self.model.can_generate() {
                        self.autogen = false;
                        self.start(cx);
                    }
                    self.request_hf();
                    if reload {
                        self.reload_history(cx);
                    }
                }
                "phase" => self.phase = v["phase"].as_str().unwrap_or("off").into(),
                "started" => {
                    self.pending = false;
                    self.running = true;
                    self.started = Some(Instant::now());
                    self.progress = (0, 0, 0, 0);
                    self.drop_live();
                    self.job_results.clear();
                    self.say(tr!("Generando…", "Generating…"), Level::Info);
                }
                "progress" => {
                    let g = |k: &str| v[k].as_u64().unwrap_or(0) as u32;
                    self.progress = (g("step"), g("total"), g("index"), g("count"));
                }
                "done" => {
                    self.pending = false;
                    self.running = false;
                    self.drop_live();
                    let msg = v["message"].as_str().unwrap_or("").to_string();
                    match v["status"].as_str() {
                        Some("error") => self.say(msg, Level::Error),
                        Some("stopped") => self.say(msg, Level::Info),
                        _ => self.say(format!("{} · {msg}", tr!("Listo", "Done")), Level::Ok),
                    }
                }
                "toast" => {
                    self.pending = false;
                    let lvl = match v["level"].as_str() {
                        Some("error") => Level::Error,
                        Some("ok") => Level::Ok,
                        _ => Level::Info,
                    };
                    self.say(v["message"].as_str().unwrap_or(""), lvl);
                }
                _ => {}
            },
            Msg::Preview(tex) => {
                self.aurora.push(tex);
            }
            Msg::Result(item) => {
                let png = item.png.clone();
                self.job_results.push(png.clone());
                self.history.insert(0, item);
                self.select(Some(png));
            }
            Msg::Prepared { source, sprite } => {
                if let Some(s) = self.source.as_mut().filter(|s| s.path == source) {
                    s.sprite = Some(sprite);
                }
            }
            Msg::Exited(tail) => {
                self.running = false;
                self.pending = false;
                self.phase = "off".into();
                let log = self.backend.take().map(|b| crate::settings::display_path(&b.log)).unwrap_or_default();
                self.say(
                    tr!(
                        format!("El motor se ha cerrado. Reinicia Dotsmith. Registro: {log}\n{}", tail.trim()),
                        format!("The engine stopped. Restart Dotsmith. Log: {log}\n{}", tail.trim())
                    ),
                    Level::Error,
                );
            }
        }
        let _ = cx;
    }

    /// Ask Hugging Face about every repo the catalog mentions: the family cards and the download sources.
    fn request_hf(&self) {
        let token = self.settings.hf_token();
        let mut by_repo: HashMap<String, Vec<String>> = HashMap::new();
        for repo in self.model.families.iter().map(|f| &f.repo).chain(self.model.store.iter().map(|s| &s.0)) {
            by_repo.entry(repo.clone()).or_default();
        }
        for r in self.model.files.iter().flat_map(|f| f.1.iter()).filter_map(|r| r.remote.as_ref()) {
            by_repo.entry(r.repo.clone()).or_default().push(r.path.clone());
        }
        for (repo, paths) in by_repo {
            self.hf.request(&repo, paths, token.clone());
        }
    }

    fn hf_info(&self, repo: &str) -> Option<(Arc<crate::hf::RepoInfo>, Option<Arc<RenderImage>>)> {
        match self.hf.get(repo)? {
            crate::hf::State::Ready(i, a) => Some((i, a)),
            _ => None,
        }
    }

    /// Size of a downloadable file, from Hugging Face.
    fn remote_size(&self, r: &RemoteFile) -> Option<u64> {
        self.hf_info(&r.repo).and_then(|(i, _)| i.sizes.get(&r.path).copied())
    }

    fn drop_live(&mut self) {
        if let Some(old) = self.aurora.clear() {
            self.trash.push(old);
        }
    }

    /// Read the output folder on a background thread; results that arrive meanwhile stay on top.
    fn reload_history(&mut self, cx: &mut Context<Self>) {
        for it in self.history.drain(..) {
            self.trash.push(it.thumb);
        }
        self.select(None);
        let dir = self.model.out_dir.clone();
        // results parked for an undo by a session that ended without emptying them go to the Recycle Bin now
        let park = dir.join(".dotsmith-trash");
        let pending = self.deleted.as_ref().is_some_and(|d| d.0.iter().any(|f| f.1.starts_with(&park)));
        if !pending {
            let left: Vec<PathBuf> = std::fs::read_dir(&park).into_iter().flatten().flatten().map(|e| e.path()).collect();
            if !left.is_empty() {
                let _ = crate::pixels::recycle(&left);
            }
            let _ = std::fs::remove_dir(&park);
        }
        cx.spawn(async move |this, cx| {
            let scan_dir = dir.clone();
            let items = cx.background_executor().spawn(async move { pixels::scan(&scan_dir, 400) }).await;
            let _ = this.update(cx, |s, cx| {
                if s.model.out_dir != dir {
                    s.trash.extend(items.into_iter().map(|i| i.thumb));
                    return;
                }
                let known: Vec<PathBuf> = s.history.iter().map(|i| i.png.clone()).collect();
                for it in items {
                    if known.contains(&it.png) {
                        s.trash.push(it.thumb);
                    } else {
                        s.history.push(it);
                    }
                }
                if s.selected.is_none() {
                    let first = s.history.first().map(|i| i.png.clone());
                    s.select(first);
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn select(&mut self, png: Option<PathBuf>) {
        self.big = None;
        if let Some((a, b, _, _)) = self.small.take() {
            self.trash.push(a);
            self.trash.push(b);
        }
        let item = png.as_ref().and_then(|p| self.history.iter().find(|i| &i.png == p));
        self.big = item.map(|i| Arc::new(i.sprite.clone()));
        self.small = item.map(|i| {
            let (w, h) = i.sprite.dimensions();
            (pixels::render(&i.sprite, 1), pixels::render(&i.sprite, 2), w, h)
        });
        self.selected = png;
    }

    fn selected_item(&self) -> Option<&Item> {
        let p = self.selected.as_ref()?;
        self.history.iter().find(|i| &i.png == p)
    }

    // ------------------------------------------------------------ actions
    fn generate(&mut self, _: &Generate, _: &mut Window, cx: &mut Context<Self>) {
        self.start(cx);
    }
    fn copy_image_action(&mut self, _: &CopyImage, _: &mut Window, cx: &mut Context<Self>) {
        self.copy_image(false);
        cx.notify();
    }
    /// The selected result to the clipboard: real pixels, or (`big`) enlarged with nearest neighbour to >= 512 px so a
    /// 32 px sprite is visible when pasted into a chat or document.
    fn copy_image(&mut self, big: bool) {
        let Some(im) = self.selected_item().map(|i| i.sprite.clone()) else {
            self.say(tr!("No hay ninguna imagen seleccionada.", "No image selected."), Level::Info);
            return;
        };
        let (w, h) = im.dimensions();
        let k = if big { (512 / w.max(h).max(1)).max(1) } else { 1 };
        let im = if k > 1 { image::imageops::resize(&im, w * k, h * k, image::imageops::FilterType::Nearest) } else { im };
        match crate::clipboard::copy_image(&im) {
            Ok(()) => self.say(
                format!(
                    "{} · {}×{} px{}",
                    tr!("Imagen copiada", "Image copied"),
                    im.width(),
                    im.height(),
                    if k > 1 { format!(" (×{k})") } else { String::new() }
                ),
                Level::Ok,
            ),
            Err(e) => self.say(tr!(format!("No se pudo copiar la imagen: {e}"), format!("Could not copy the image: {e}")), Level::Error),
        }
    }
    fn stop(&mut self, _: &Stop, window: &mut Window, cx: &mut Context<Self>) {
        if self.palette_open {
            self.close_palette(window, cx);
        } else if self.settings_open || self.model_open || self.hub_open || self.downloads_open || self.menu_open {
            self.menu_open = false;
            self.settings_open = false;
            self.model_open = false;
            self.hub_open = false;
            self.downloads_open = false;
            cx.notify();
        } else if self.running {
            self.send(json!({"cmd": "stop"}));
            self.say(tr!("Deteniendo en el siguiente paso…", "Stopping at the next step…"), Level::Info);
        }
    }
    fn paste_image(&mut self, _: &PasteImage, _: &mut Window, cx: &mut Context<Self>) {
        let Some(item) = cx.read_from_clipboard() else { return };
        for e in item.entries() {
            if let ClipboardEntry::Image(im) = e {
                match image::load_from_memory(&im.bytes) {
                    Ok(dec) => {
                        let dir = std::env::temp_dir().join("sprite-studio-paste");
                        let _ = std::fs::create_dir_all(&dir);
                        let path = dir.join(format!(
                            "paste_{}.png",
                            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
                        ));
                        if dec.to_rgba8().save(&path).is_ok() {
                            if !self.mode.uses_source() {
                                self.mode = Mode::Variants;
                            }
                            self.set_source(path, cx);
                        }
                    }
                    Err(e) => self.say(tr!(format!("No se pudo leer la imagen pegada: {e}"), format!("Could not read the pasted image: {e}")), Level::Error),
                }
                cx.notify();
                return;
            }
        }
    }

    fn start(&mut self, cx: &mut Context<Self>) {
        if self.running || self.pending {
            return;
        }
        if self.mode != Mode::Pose && self.model.gpu.is_some() && !self.model.can_generate() {
            self.settings_tab = if self.model.gpu.as_ref().is_some_and(|g| !g.ok) { settings_view::Tab::Gpu } else { settings_view::Tab::Models };
            self.settings_open = true;
            self.say(
                tr!("Falta configurar Dotsmith: revisa la GPU y los archivos del modelo.", "Dotsmith is not set up yet: check the GPU and the model files."),
                Level::Error,
            );
            cx.notify();
            return;
        }
        if self.mode == Mode::Pose {
            self.save_pose(false, cx);
            return;
        }
        if !self.mode_ok(self.mode) {
            let name = self.family_info().map(|f| f.label.clone()).unwrap_or_default();
            self.say(
                tr!(format!("{name} no hace este modo. Elige otro modelo (Ctrl+L)."), format!("{name} can't do this mode. Pick another model (Ctrl+L).")),
                Level::Error,
            );
            cx.notify();
            return;
        }
        let mode = self.mode;
        let f = self.form();
        let prompt = f.prompt.read(cx).text().trim().to_string();
        let seed_text = f.seed.read(cx).text().trim().to_string();
        let seed: i64 = if seed_text.is_empty() {
            -1
        } else {
            match seed_text.parse() {
                Ok(s) => s,
                Err(_) => {
                    self.say(tr!("La semilla debe ser un número (vacía = aleatoria).", "The seed must be a number (empty = random)."), Level::Error);
                    cx.notify();
                    return;
                }
            }
        };
        let mut req = json!({"cmd": "generate", "mode": mode.key(), "prompt": prompt, "seed": seed, "steps": self.steps_of(f),
                             "camera": CAMERAS[f.camera].1, "lora_scale": self.lora_scale});
        match mode {
            Mode::Single => {
                let (_, mut w, mut h) = SINGLE_SIZES[f.size];
                if !self.size_ok(w, h) {
                    // the nearest size this model draws
                    let ok = SINGLE_SIZES.iter().filter(|s| self.size_ok(s.1, s.2)).min_by_key(|s| s.1.abs_diff(w));
                    if let Some(s) = ok {
                        (w, h) = (s.1, s.2);
                    }
                }
                req["width"] = w.into();
                req["height"] = h.into();
                req["opaque"] = f.opaque.into();
                req["center"] = f.center.into();
                req["mirror"] = f.mirror.into();
            }
            Mode::Pack => {
                let (_, k, c) = PACKS[f.size];
                req["grid"] = k.into();
                req["cell"] = c.into();
                req["opaque"] = f.opaque.into();
                req["center"] = f.center.into();
            }
            Mode::Creature => self.creature_request(&mut req),
            Mode::Walls => req["room"] = f.room.into(),
            _ => {
                let Some(src) = self.source.as_ref() else {
                    self.say(
                        tr!("Carga primero una imagen de origen (abrir, arrastrar o Ctrl+V).", "Load a source image first (open, drag or Ctrl+V)."),
                        Level::Error,
                    );
                    cx.notify();
                    return;
                };
                req["source"] = src.path.to_string_lossy().to_string().into();
                req["strength"] = f.strength.into();
                req["variants"] = f.variants.into();
                req["guidance"] = f.guidance.into();
                req["out_size"] = OUT_SIZES[f.size.min(2)].into();
                if mode == Mode::Photo {
                    req["opaque"] = f.opaque.into();
                    req["restyle"] = f.restyle.into();
                    req["out_size"] = DIRECT_SIZES[f.size.min(4)].into();
                }
                if mode == Mode::Regions {
                    if !self.mask.iter().any(|&b| b) {
                        self.say(tr!("Pinta la zona editable antes de generar.", "Paint the editable area before generating."), Level::Error);
                        cx.notify();
                        return;
                    }
                    req["mask"] = self.mask.iter().map(|&b| if b { '1' } else { '0' }).collect::<String>().into();
                }
                if mode == Mode::Photo {
                    req["camera"] = "Normal".into();
                }
                if mode == Mode::Angle {
                    req["pitch"] = f.pitch.into();
                    req["yaw"] = f.yaw.into();
                    req["warp"] = f.warp.into();
                }
            }
        }
        self.pending = true;
        self.send(req);
        self.say(
            if self.phase == "off" { tr!("Cargando el modelo en la GPU…", "Loading the model on the GPU…") } else { tr!("Enviando…", "Sending…") },
            Level::Info,
        );
        cx.notify();
    }

    fn set_source(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        match pixels::load_source(&path) {
            Some((view, w, h)) => {
                if let Some(old) = self.source.take() {
                    self.trash.push(old.view);
                }
                self.source = Some(Source { path: path.clone(), view, w, h, sprite: None });
                self.mask = vec![false; 1024];
                self.send(json!({"cmd": "prepare", "path": path.to_string_lossy()}));
            }
            None => self.say(tr!(format!("No se pudo abrir {}", path.display()), format!("Could not open {}", path.display())), Level::Error),
        }
        cx.notify();
    }

    fn open_source(&mut self, cx: &mut Context<Self>) {
        let rx =
            cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some(tr!("Abrir imagen", "Open image").into()) });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                let _ = this.update(cx, |s, cx| s.set_source(p, cx));
            }
        })
        .detach();
    }

    /// Put a past result's settings back into the controls of its mode.
    fn reuse(&mut self, cx: &mut Context<Self>) {
        let Some(item) = self.selected_item() else { return };
        let j = item.json.clone();
        let Some(mode) = j["mode"].as_str().and_then(Mode::from_key) else { return };
        self.mode = mode;
        let s = |k: &str| j[k].as_str().unwrap_or("").to_string();
        let prompt = s("prompt");
        let seed = j["seed"].as_i64().map(|v| v.to_string()).unwrap_or_default();
        if let Some(v) = j["lora_scale"].as_f64() {
            self.lora_scale = v as f32;
        }
        let f = &mut self.forms[mode as usize];
        f.prompt.update(cx, |t, cx| t.set_text(&prompt, cx));
        f.seed.update(cx, |t, cx| t.set_text(&seed, cx));
        f.camera = camera_index(&s("camera"));
        f.opaque = j["opaque"].as_bool().unwrap_or(false);
        f.center = j["center"].as_bool().unwrap_or(false);
        f.mirror = j["mirror"].as_bool().unwrap_or(false);
        if let Some(st) = j["steps"].as_u64().map(|v| v as u32) {
            if j["base"].as_str().unwrap_or("").contains("Turbo") { f.turbo_steps = Some(st) } else { f.steps = Some(st) }
        }
        if let Some(v) = j["strength"].as_f64() {
            f.strength = v as f32;
        }
        if let Some(v) = j["guidance"].as_f64() {
            f.guidance = v as f32;
        }
        if let Some(v) = j["variants"].as_u64() {
            f.variants = v as u32;
        }
        if let Some(v) = j["pitch"].as_f64() {
            f.pitch = v as f32;
        }
        if let Some(v) = j["yaw"].as_f64() {
            f.yaw = v as f32;
        }
        if let Some(v) = j["warp"].as_bool() {
            f.warp = v;
        }
        if let Some(v) = j["room"].as_bool() {
            f.room = v;
        }
        match mode {
            Mode::Single => {
                let (w, h) = (j["width"].as_u64().unwrap_or(32) as u32, j["height"].as_u64().unwrap_or(32) as u32);
                f.size = SINGLE_SIZES.iter().position(|x| x.1 == w && x.2 == h).unwrap_or(1);
            }
            Mode::Pack => {
                let (k, c) = (j["grid"].as_u64().unwrap_or(3) as u32, j["cell"].as_u64().unwrap_or(32) as u32);
                f.size = PACKS.iter().position(|x| x.1 == k && x.2 == c).unwrap_or(0);
            }
            Mode::Creature => {
                f.camera = crate::jobs::CREATURE_CAMERAS.iter().position(|c| j["camera"].as_str() == Some(c)).unwrap_or(0);
                let (w, h) = (j["cell"].as_u64().unwrap_or(32) as u32, j["cell_h"].as_u64().or(j["cell"].as_u64()).unwrap_or(32) as u32);
                f.size = creature_mode::CELLS[f.camera].iter().position(|&c| c == (w, h)).unwrap_or(0);
                if let Some(a) = j["dirs"].as_array() {
                    f.all_dirs = a.len() == 4;
                }
            }
            Mode::Photo => {
                f.restyle = j["restyle"].as_bool().unwrap_or(false);
                f.size = DIRECT_SIZES.iter().position(|&x| x as u64 == j["out_size"].as_u64().unwrap_or(64)).unwrap_or(2);
            }
            _ => {
                f.size = OUT_SIZES.iter().position(|&x| x as u64 == j["out_size"].as_u64().unwrap_or(32)).unwrap_or(0);
            }
        }
        if mode.uses_source() {
            let src = PathBuf::from(s("source"));
            if src.is_file() {
                self.set_source(src, cx);
                if let Some(m) = j["mask"].as_str().filter(|m| m.len() == 1024) {
                    self.mask = m.chars().map(|c| c == '1').collect();
                }
            } else {
                self.say(tr!("La imagen de origen ya no existe; carga otra.", "The source image no longer exists; load another one."), Level::Info);
            }
        }
        cx.notify();
    }

    fn open_prompt_modal(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.prompt_modal = true;
        let p = self.form().prompt.clone();
        p.update(cx, |t, cx| t.set_lines(MODAL_LINES.0, MODAL_LINES.1, cx));
        window.focus(&p.read(cx).focus_handle.clone());
        cx.notify();
    }

    fn close_prompt_modal(&mut self, _: &ClosePromptModal, window: &mut Window, cx: &mut Context<Self>) {
        if !self.prompt_modal {
            return;
        }
        self.prompt_modal = false;
        let mode = self.mode;
        let p = self.form().prompt.clone();
        let (min, max) = if mode == Mode::Pack { (6, 11) } else { PANEL_LINES };
        p.update(cx, |t, cx| t.set_lines(min, max, cx));
        window.focus(&p.read(cx).focus_handle.clone());
        cx.notify();
    }

    /// The layer every modal sits on: a dimmed backdrop that closes it on click, the card centred, all fading in.
    pub(super) fn modal_layer(
        &mut self,
        id: &'static str,
        card: impl IntoElement,
        close: fn(&mut Studio, &mut Window, &mut Context<Studio>),
        cx: &mut Context<Self>,
    ) -> AnyElement {
        use gpui::{Animation, AnimationExt};
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .p(px(24.))
            .child(div().id(id).absolute().top_0().left_0().size_full().bg(rgba(VEIL())).on_click(cx.listener(move |this, _, w, cx| {
                close(this, w, cx);
                cx.notify();
            })))
            .child(card)
            .with_animation(id, Animation::new(Duration::from_millis(140)).with_easing(gpui::ease_out_quint()), |d, t| d.opacity(t))
            .into_any_element()
    }

    fn prompt_modal_view(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let prompt = self.form().prompt.clone();
        let mode = self.mode;
        let card = div()
            .id("modal-card")
            .occlude()
            .w(px(820.))
            .max_w_full()
            .rounded(px(14.))
            .bg(rgb(PANEL()))
            .border_1()
            .border_color(rgb(BORDER()))
            .shadow_lg()
            .flex()
            .flex_col()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .px(px(20.))
                    .pt(px(16.))
                    .pb(px(10.))
                    .child(div().flex_1().text_size(px(15.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(if mode == Mode::Pack {
                        tr!("Dirección del pack", "Pack direction")
                    } else {
                        tr!("Descripción", "Description")
                    }))
                    .child(icon_btn("modal-close", "x").on_click(cx.listener(|this, _, w, cx| this.close_prompt_modal(&ClosePromptModal, w, cx)))),
            )
            .child(div().px(px(20.)).child(prompt))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .px(px(20.))
                    .py(px(14.))
                    .child(div().flex_1())
                    .child(kbd("Esc"))
                    .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).mr(px(8.)).child(tr!("cerrar", "close")))
                    .child(kbd("Ctrl + ↵"))
                    .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).mr(px(8.)).child(tr!("generar", "generate")))
                    .child(
                        div()
                            .kb("modal-done")
                            .px(px(16.))
                            .h(px(32.))
                            .rounded(px(8.))
                            .bg(rgb(ACCENT()))
                            .flex()
                            .items_center()
                            .text_size(px(13.))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(rgb(ACCENT_INK()))
                            .cursor_pointer()
                            .hover(|s| s.bg(rgb(ACCENT_HOVER())))
                            .child(tr!("Listo", "Done"))
                            .on_click(cx.listener(|this, _, w, cx| this.close_prompt_modal(&ClosePromptModal, w, cx))),
                    ),
            );
        let layer = self.modal_layer("modal-veil", card, |s, w, cx| s.close_prompt_modal(&ClosePromptModal, w, cx), cx);
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .key_context("PromptModal")
            .on_action(cx.listener(Self::close_prompt_modal))
            .child(layer)
            .into_any_element()
    }

    fn use_as_source(&mut self, cx: &mut Context<Self>) {
        let Some(p) = self.selected.clone() else { return };
        if !self.mode.uses_source() {
            self.mode = Mode::Variants;
        }
        self.set_source(p, cx);
    }

    /// The pending delete goes to the Recycle Bin for good.
    fn finish_delete(&mut self) {
        if let Some((files, _, _)) = self.deleted.take() {
            let parked: Vec<PathBuf> = files.iter().map(|f| f.1.clone()).filter(|p| p.exists()).collect();
            let _ = crate::pixels::recycle(&parked);
            if let Some(dir) = files.first().and_then(|f| f.1.parent()) {
                let _ = std::fs::remove_dir(dir); // only when empty
            }
        }
    }

    fn undo_delete(&mut self, cx: &mut Context<Self>) {
        let Some((files, ix, _)) = self.deleted.take() else { return };
        for (orig, parked) in &files {
            let _ = std::fs::rename(parked, orig);
        }
        let png = files[0].0.clone();
        let json = std::fs::read_to_string(png.with_extension("json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
        if let Some(item) = pixels::Item::load(png.clone(), json) {
            self.history.insert(ix.min(self.history.len()), item);
            self.select(Some(png));
        }
        self.say(tr!("Recuperado.", "Restored."), Level::Info);
        cx.notify();
    }

    fn delete_selected(&mut self, cx: &mut Context<Self>) {
        let Some(p) = self.selected.clone() else { return };
        self.finish_delete();
        // parked next to the results first, so an undo is a rename; the Recycle Bin gets it after UNDO_SECS
        let park = p.parent().map(|d| d.join(".dotsmith-trash")).unwrap_or_default();
        let files: Vec<(PathBuf, PathBuf)> =
            [p.clone(), p.with_extension("json")].into_iter().filter(|f| f.exists()).filter_map(|f| Some((f.clone(), park.join(f.file_name()?)))).collect();
        let parked = std::fs::create_dir_all(&park).is_ok() && files.iter().all(|(a, b)| std::fs::rename(a, b).is_ok());
        if !parked {
            for (a, b) in &files {
                let _ = std::fs::rename(b, a);
            }
            if let Err(e) = crate::pixels::recycle(&[p.clone(), p.with_extension("json")]) {
                self.say(tr!(format!("No se pudo eliminar: {e}"), format!("Could not delete: {e}")), Level::Error);
                cx.notify();
                return;
            }
        }
        let ix = self.history.iter().position(|i| i.png == p).unwrap_or(0);
        if let Some(pos) = self.history.iter().position(|i| i.png == p) {
            let it = self.history.remove(pos);
            self.trash.push(it.thumb);
        }
        self.job_results.retain(|x| x != &p);
        let next = self.visible_history().get(ix.min(self.history.len().saturating_sub(1))).map(|i| i.png.clone());
        self.select(next);
        if parked {
            self.deleted = Some((files, ix, Instant::now()));
        }
        self.say(tr!("Movido a la papelera.", "Moved to the Recycle Bin."), Level::Info);
        cx.notify();
    }

    fn step_selection(&mut self, d: i32, cx: &mut Context<Self>) {
        let list: Vec<PathBuf> = self.visible_history().iter().map(|i| i.png.clone()).collect();
        if list.is_empty() {
            return;
        }
        let cur = self.selected.as_ref().and_then(|s| list.iter().position(|p| p == s));
        let next = match cur {
            Some(i) => (i as i32 + d).clamp(0, list.len() as i32 - 1) as usize,
            None => 0,
        };
        self.select(Some(list[next].clone()));
        cx.notify();
    }

    fn visible_history(&self) -> Vec<&Item> {
        let q = palette::fold(&self.hist_query_text);
        self.history
            .iter()
            .filter(|i| !self.only_mode || i.json["mode"].as_str() == Some(self.mode.key()))
            .filter(|i| q.is_empty() || palette::fold(i.json["prompt"].as_str().unwrap_or("")).contains(q.trim()))
            .collect()
    }

    // ------------------------------------------------------------ region mask
    fn cell_at(&self, pos: Point<Pixels>) -> Option<(i32, i32)> {
        let b = self.grid.get()?;
        let cw = b.size.width / 32.;
        let x = ((pos.x - b.origin.x) / cw).floor() as i32;
        let y = ((pos.y - b.origin.y) / cw).floor() as i32;
        ((0..32).contains(&x) && (0..32).contains(&y)).then_some((x, y))
    }
    fn paint_cell(&mut self, x: i32, y: i32, on: bool) {
        let r = self.brush - 1;
        for dy in -r..=r {
            for dx in -r..=r {
                let (cx, cy) = (x + dx, y + dy);
                if (0..32).contains(&cx) && (0..32).contains(&cy) {
                    self.mask[(cy * 32 + cx) as usize] = on;
                }
            }
        }
    }
    fn wand(&mut self, x: i32, y: i32) {
        let Some(sprite) = self.source.as_ref().and_then(|s| s.sprite.as_ref()) else { return };
        let p = sprite.get_pixel(x as u32, y as u32).0;
        if p[3] == 0 {
            return;
        }
        let (h0, s0) = pixels::hue_sat(p[0], p[1], p[2]);
        for (i, q) in sprite.pixels().enumerate() {
            let q = q.0;
            if q[3] == 0 {
                continue;
            }
            let hit = if s0 > 0.22 {
                let (h, s) = pixels::hue_sat(q[0], q[1], q[2]);
                let d = (h - h0).abs();
                s > 0.22 && d.min(1. - d) < 0.09
            } else {
                (0..3).all(|c| (q[c] as i32 - p[c] as i32).abs() < 40)
            };
            if hit {
                self.mask[i] = true;
            }
        }
    }
    fn on_grid_down(&mut self, ev: &MouseDownEvent, erase: bool, cx: &mut Context<Self>) {
        let Some((x, y)) = self.cell_at(ev.position) else { return };
        if self.tool == Tool::Wand && !erase {
            self.wand(x, y);
        } else {
            let on = !(erase || self.tool == Tool::Erase);
            self.painting = Some(on);
            self.paint_cell(x, y, on);
        }
        cx.notify();
    }
    fn on_grid_move(&mut self, ev: &MouseMoveEvent, cx: &mut Context<Self>) {
        if let (Some(on), Some((x, y))) = (self.painting, self.cell_at(ev.position)) {
            self.paint_cell(x, y, on);
            cx.notify();
        }
    }
}

// ================================================================ angle gizmo
const GIZMO_PRESETS: [(f32, f32); 6] = [(0., 0.), (30., 0.), (45., 0.), (35., 40.), (15., 90.), (88., 0.)];

fn gizmo_preset_label(i: usize) -> &'static str {
    match i {
        0 => tr!("Frente", "Front"),
        1 => tr!("Oblicua", "Oblique"),
        2 => tr!("Cenital 45°", "Top-down 45°"),
        3 => "3/4",
        4 => tr!("Perfil", "Profile"),
        _ => tr!("Planta", "Top"),
    }
}

fn paint_poly(pts: &[Point<Pixels>], color: u32, alpha: f32, stroke: Option<f32>, window: &mut Window) {
    let mut pb = match stroke {
        Some(w) => gpui::PathBuilder::stroke(px(w)),
        None => gpui::PathBuilder::fill(),
    };
    pb.move_to(pts[0]);
    for q in &pts[1..] {
        pb.line_to(*q);
    }
    pb.close();
    if let Ok(path) = pb.build() {
        let mut col: gpui::Hsla = rgb(color).into();
        col.a = alpha;
        window.paint_path(path, col);
    }
}

/// A shaded cube seen from (pitch, yaw) over a ground ring with a front marker: what the camera will see.
fn paint_gizmo(b: Bounds<Pixels>, pitch: f32, yaw: f32, window: &mut Window) {
    let (p, y) = (pitch.to_radians(), yaw.to_radians());
    let c = point(b.origin.x + b.size.width / 2., b.origin.y + b.size.height * 0.54);
    let s = f32::from(b.size.height) * 0.2;
    // object space: x right, y up, z toward the viewer at yaw 0
    let rot = move |v: [f32; 3]| -> [f32; 3] {
        let (x, z) = (v[0] * y.cos() - v[2] * y.sin(), v[0] * y.sin() + v[2] * y.cos());
        [x, v[1] * p.cos() - z * p.sin(), v[1] * p.sin() + z * p.cos()] // camera above: the top faces it
    };
    let scr = move |v: [f32; 3]| {
        let r = rot(v);
        point(c.x + px(r[0] * s), c.y - px(r[1] * s))
    };
    let ring: Vec<Point<Pixels>> = (0..48)
        .map(|i| {
            let t = i as f32 / 48. * std::f32::consts::TAU;
            scr([1.7 * t.cos(), -1., 1.7 * t.sin()])
        })
        .collect();
    paint_poly(&ring, BORDER(), 1., Some(1.), window);
    let front = scr([0., -1., 1.95]);
    window.paint_quad(fill(Bounds::new(point(front.x - px(3.), front.y - px(3.)), size(px(6.), px(6.))), rgb(BLUE())).corner_radii(px(3.)));
    let faces: [([f32; 3], [[f32; 3]; 4], u32); 6] = [
        ([0., 1., 0.], [[-1., 1., -1.], [1., 1., -1.], [1., 1., 1.], [-1., 1., 1.]], 0xe8e8e8),
        ([0., 0., 1.], [[-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]], BLUE()),
        ([1., 0., 0.], [[1., -1., 1.], [1., -1., -1.], [1., 1., -1.], [1., 1., 1.]], 0x5a5a5a),
        ([-1., 0., 0.], [[-1., -1., -1.], [-1., -1., 1.], [-1., 1., 1.], [-1., 1., -1.]], 0x5a5a5a),
        ([0., 0., -1.], [[1., -1., -1.], [-1., -1., -1.], [-1., 1., -1.], [1., 1., -1.]], 0x3a3a3a),
        ([0., -1., 0.], [[-1., -1., 1.], [1., -1., 1.], [1., -1., -1.], [-1., -1., -1.]], 0x222222),
    ];
    for (n, corners, shade) in faces {
        let depth = rot(n)[2];
        if depth <= 0.01 {
            continue;
        }
        let pts: Vec<Point<Pixels>> = corners.iter().map(|&v| scr(v)).collect();
        paint_poly(&pts, shade, 0.35 + 0.55 * depth.min(1.), None, window);
        paint_poly(&pts, TEXT(), 0.9, Some(1.2), window);
    }
}

impl Studio {
    fn gizmo_panel(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let (pitch, yaw, warp) = (self.form().pitch, self.form().yaw, self.form().warp);
        let pad = div()
            .id("gizmo")
            .relative()
            .w_full()
            .h(px(250.))
            .rounded(px(10.))
            .bg(rgb(INPUT_BG()))
            .border_1()
            .border_color(rgb(if self.gizmo_drag.is_some() { ACCENT_DIM() } else { BORDER() }))
            .cursor_grab()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, ev: &MouseDownEvent, _, cx| {
                    if ev.click_count == 2 {
                        let f = this.form_mut();
                        f.pitch = 45.;
                        f.yaw = 0.;
                        this.gizmo_drag = None;
                    } else {
                        let (p, y) = (this.form().pitch, this.form().yaw);
                        this.gizmo_drag = Some((ev.position, p, y));
                    }
                    cx.notify();
                }),
            )
            .on_mouse_move(cx.listener(|this, ev: &MouseMoveEvent, _, cx| {
                if ev.pressed_button != Some(MouseButton::Left) {
                    this.gizmo_drag = None;
                    return;
                }
                if let Some((start, p0, y0)) = this.gizmo_drag {
                    let dx = f32::from(ev.position.x - start.x);
                    let dy = f32::from(ev.position.y - start.y);
                    // you hold the cube: drag right and its front turns right, drag down and its top tips toward you
                    let free = ev.modifiers.shift;
                    let snap = |v: f32, step: f32| if free { v } else { (v / step).round() * step };
                    let f = this.form_mut();
                    f.yaw = snap((y0 - dx * 0.5 + 180.).rem_euclid(360.) - 180., 15.);
                    f.pitch = snap((p0 + dy * 0.4).clamp(0., 90.), 5.).clamp(0., 90.);
                    cx.notify();
                }
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    this.gizmo_drag = None;
                    cx.notify();
                }),
            )
            .on_mouse_up_out(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    this.gizmo_drag = None;
                    cx.notify();
                }),
            )
            .child(canvas(|_, _, _| (), move |b, _, window, _| paint_gizmo(b, pitch, yaw, window)).size_full())
            .child(
                div()
                    .absolute()
                    .top(px(10.))
                    .left_0()
                    .right_0()
                    .flex()
                    .justify_center()
                    .text_size(px(10.5))
                    .text_color(rgb(MUTED_2()))
                    .child(tr!("Arrastra para girar · doble clic restablece", "Drag to rotate · double-click resets")),
            );
        div()
            .flex()
            .flex_col()
            .gap(px(8.))
            .child(pad)
            .child(div().flex().flex_wrap().gap(px(6.)).children(GIZMO_PRESETS.iter().enumerate().map(|(i, (p, y))| {
                let (p, y) = (*p, *y);
                let name = gizmo_preset_label(i);
                let active = (pitch - p).abs() < 0.5 && (yaw - y).abs() < 0.5;
                chip(("gz", i), name, active).on_click(cx.listener(move |this, _, _, cx| {
                    let f = this.form_mut();
                    f.pitch = p;
                    f.yaw = y;
                    cx.notify();
                }))
            })))
            .child(self.stepper("pitch", tr!("Inclinación", "Pitch"), format!("{pitch:.0}°"), "", |f, d| f.pitch = (f.pitch + d * 5.).clamp(0., 90.), cx))
            .child(self.stepper("yaw", tr!("Giro", "Yaw"), format!("{yaw:.0}°"), "", |f, d| f.yaw = (f.yaw + d * 15. + 180.).rem_euclid(360.) - 180., cx))
            .child(
                toggle(
                    "warp",
                    tr!("Deformar el origen hacia el ángulo", "Warp the source toward the angle"),
                    tr!("Desactívalo si el origen no está de frente.", "Turn it off if the source isn't seen from the front."),
                    warp,
                )
                .on_click(cx.listener(|this, _, _, cx| {
                    let f = this.form_mut();
                    f.warp = !f.warp;
                    cx.notify();
                })),
            )
            .into_any_element()
    }
}

/// A history result being dragged onto the left panel.
#[derive(Clone)]
pub struct DraggedSprite {
    png: PathBuf,
    thumb: Arc<RenderImage>,
}

struct DragPreview {
    thumb: Arc<RenderImage>,
}

impl Render for DragPreview {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(64.))
            .h(px(64.))
            .rounded(px(8.))
            .bg(rgb(RAISED()))
            .border_2()
            .border_color(rgb(BLUE()))
            .opacity(0.92)
            .p(px(4.))
            .child(img(self.thumb.clone()).size_full().object_fit(ObjectFit::Contain))
    }
}

/// Start the system window move (as if the user grabbed a native caption).
#[cfg(windows)]
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

#[cfg(not(windows))]
fn begin_native_move(_: &Window) {}

// ================================================================ widgets
fn label(text: impl Into<SharedString>) -> Div {
    div().text_size(px(10.5)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(MUTED())).child(text.into())
}

fn section(title: &str) -> Div {
    div()
        .flex()
        .items_center()
        .gap_2()
        .pt(px(18.))
        .pb(px(8.))
        .child(div().text_size(px(12.5)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(title.to_string()))
        .child(div().flex_1().h(px(1.)).bg(rgb(BORDER_SOFT())))
}

fn chip(id: impl Into<gpui::ElementId>, text: &str, active: bool) -> Stateful<Div> {
    div()
        .kb(id.into())
        .px(px(10.))
        .h(px(28.))
        .flex()
        .items_center()
        .justify_center()
        .rounded(px(6.))
        .text_size(px(12.))
        .cursor_pointer()
        .when(active, |d| d.bg(rgb(ACCENT())).text_color(rgb(ACCENT_INK())).font_weight(FontWeight::SEMIBOLD))
        .when(!active, |d| d.bg(rgb(RAISED())).text_color(rgb(TEXT())).border_1().border_color(rgb(BORDER())).hover(|s| s.bg(rgb(HOVER()))))
        .child(text.to_string())
}

/// A switch with its label; `hint` is shown on hover behind a small info mark, never as a line of text.
fn toggle(id: impl Into<gpui::ElementId>, text: &str, hint: &str, on: bool) -> Stateful<Div> {
    div()
        .kb(id.into())
        .flex()
        .items_center()
        .gap(px(10.))
        .py(px(6.))
        .cursor_pointer()
        .hover(|s| s.opacity(0.9))
        .when(!hint.is_empty(), |d| d.tooltip(tooltip(hint)))
        .child(switch(on))
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap(px(6.))
                .child(div().text_size(px(12.5)).text_color(rgb(TEXT())).child(text.to_string()))
                .when(!hint.is_empty(), |d| d.child(info_mark())),
        )
}

fn switch(on: bool) -> Div {
    div()
        .w(px(30.))
        .h(px(17.))
        .flex_none()
        .rounded_full()
        .p(px(2.))
        .flex()
        .when(on, |d| d.bg(rgb(ACCENT())).justify_end())
        .when(!on, |d| d.bg(rgb(HOVER())).justify_start())
        .child(div().w(px(13.)).h(px(13.)).rounded_full().bg(rgb(if on { ACCENT_INK() } else { MUTED() })))
}

fn info_mark() -> gpui::Svg {
    icon("info").size(px(12.)).text_color(rgb(MUTED_2()))
}

/// A segmented control: one pill per option, the active one raised.
fn segmented(items: impl IntoIterator<Item = Stateful<Div>>) -> Div {
    div()
        .flex()
        .flex_none()
        .items_center()
        .p(px(2.))
        .gap(px(2.))
        .rounded(px(8.))
        .bg(rgb(INPUT_BG()))
        .border_1()
        .border_color(rgb(BORDER_SOFT()))
        .children(items)
}

fn seg(id: impl Into<gpui::ElementId>, text: &str, active: bool) -> Stateful<Div> {
    div()
        .kb(id.into())
        .h(px(26.))
        .px(px(11.))
        .flex()
        .items_center()
        .justify_center()
        .gap(px(5.))
        .rounded(px(6.))
        .text_size(px(12.))
        .whitespace_nowrap()
        .cursor_pointer()
        .when(active, |d| d.bg(rgb(RAISED())).border_1().border_color(rgb(BORDER())).text_color(rgb(TEXT())).font_weight(FontWeight::SEMIBOLD))
        .when(!active, |d| d.text_color(rgb(MUTED())).hover(|s| s.text_color(rgb(TEXT())).bg(rgb(PANEL()))))
        .child(text.to_string())
}

/// A settings row in the Obsidian manner: name and a one-line description on the left, the control on the right.
fn setting_row(name: &str, desc: impl Into<SharedString>, control: impl IntoElement) -> Div {
    let desc = desc.into();
    div()
        .flex()
        .items_center()
        .gap(px(16.))
        .py(px(12.))
        .border_t_1()
        .border_color(rgb(BORDER_SOFT()))
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap(px(3.))
                .child(div().text_size(px(13.)).text_color(rgb(TEXT())).child(name.to_string()))
                .when(!desc.is_empty(), |d| d.child(div().text_size(px(11.5)).line_height(px(16.)).text_color(rgb(MUTED_2())).child(desc))),
        )
        .child(div().flex_none().flex().items_center().gap(px(6.)).child(control))
}

/// A keyboard key as drawn in menus and the shortcut list.
fn kbd(text: &str) -> Div {
    div()
        .px(px(6.))
        .h(px(20.))
        .flex()
        .items_center()
        .rounded(px(5.))
        .bg(rgb(RAISED()))
        .border_1()
        .border_color(rgb(BORDER()))
        .font_family(MONO)
        .text_size(px(10.5))
        .text_color(rgb(MUTED()))
        .child(text.to_string())
}

struct Tooltip(SharedString);

impl Render for Tooltip {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .px(px(8.))
            .py(px(4.))
            .rounded(px(6.))
            .bg(rgb(RAISED()))
            .border_1()
            .border_color(rgb(BORDER()))
            .shadow_lg()
            .font_family(FONT)
            .text_size(px(11.5))
            .text_color(rgb(TEXT()))
            .child(self.0.clone())
    }
}

/// One status bar item: same height, padding and hover for every entry, so the bar reads as one strip.
fn status_item(id: &'static str) -> Stateful<Div> {
    div().id(id).h(px(22.)).px(px(8.)).flex().items_center().gap(px(6.)).rounded(px(5.)).hover(|s| s.bg(rgb(PANEL())))
}

fn status_sep() -> Div {
    div().w(px(1.)).h(px(12.)).bg(rgb(BORDER_SOFT()))
}

/// Memory meter: label, a small fill bar and "used / total GB"; amber from 90 % full, red from 97 %.
fn mem_meter(id: &'static str, label: &'static str, used: u64, total: u64, tip: &str) -> Stateful<Div> {
    let frac = if total > 0 { (used as f64 / total as f64) as f32 } else { 0.0 };
    let col = if frac >= 0.97 {
        DANGER()
    } else if frac >= 0.9 {
        WARN()
    } else {
        MUTED()
    };
    status_item(id)
        .tooltip(tooltip(tip))
        .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(label))
        .child(
            div()
                .w(px(34.))
                .h(px(4.))
                .rounded_full()
                .bg(rgb(RAISED()))
                .overflow_hidden()
                .child(div().h_full().w(px(34. * frac.clamp(0.04, 1.))).rounded_full().bg(rgb(col))),
        )
        .child(div().font_family(MONO).text_size(px(10.5)).text_color(rgb(if col == MUTED() { MUTED() } else { col })).child(format!(
            "{:.1}/{:.0} GB",
            used as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64
        )))
}

fn tooltip(text: &str) -> impl Fn(&mut Window, &mut gpui::App) -> gpui::AnyView + 'static {
    let text = SharedString::from(text.to_string());
    move |_, cx| cx.new(|_| Tooltip(text.clone())).into()
}

/// A title-bar button: an icon with its name on hover, lit while its panel is open.
fn bar_btn(id: &'static str, name: &'static str, tip: &str, open: bool, color: u32) -> Stateful<Div> {
    div()
        .kb(id)
        .relative()
        .w(px(32.))
        .h(px(30.))
        .flex()
        .items_center()
        .justify_center()
        .rounded(px(7.))
        .cursor_pointer()
        .when(open, |d| d.bg(rgb(RAISED())))
        .hover(|s| s.bg(rgb(PANEL())))
        .tooltip(tooltip(tip))
        .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
        .child(icon(name).size(px(16.)).text_color(rgb(if open { TEXT() } else { color })))
}

fn icon_btn(id: impl Into<gpui::ElementId>, name: &'static str) -> Stateful<Div> {
    div()
        .kb(id.into())
        .w(px(26.))
        .h(px(26.))
        .flex()
        .items_center()
        .justify_center()
        .rounded(px(6.))
        .bg(rgb(RAISED()))
        .border_1()
        .border_color(rgb(BORDER()))
        .text_size(px(13.))
        .text_color(rgb(TEXT()))
        .cursor_pointer()
        .hover(|s| s.bg(rgb(HOVER())))
        .child(icon(name).size(px(14.)).text_color(rgb(TEXT())))
}

fn button(id: impl Into<gpui::ElementId>, text: &str) -> Stateful<Div> {
    div()
        .kb(id.into())
        .px(px(12.))
        .h(px(30.))
        .flex()
        .items_center()
        .justify_center()
        .gap(px(6.))
        .rounded(px(7.))
        .bg(rgb(RAISED()))
        .border_1()
        .border_color(rgb(BORDER()))
        .text_size(px(12.))
        .text_color(rgb(TEXT()))
        .cursor_pointer()
        .hover(|s| s.bg(rgb(HOVER())))
        .child(text.to_string())
}

fn button_i(id: impl Into<gpui::ElementId>, name: &'static str, text: &str) -> Stateful<Div> {
    div()
        .kb(id.into())
        .px(px(12.))
        .h(px(30.))
        .flex()
        .items_center()
        .justify_center()
        .gap(px(7.))
        .rounded(px(7.))
        .bg(rgb(RAISED()))
        .border_1()
        .border_color(rgb(BORDER()))
        .text_size(px(12.))
        .text_color(rgb(TEXT()))
        .cursor_pointer()
        .hover(|s| s.bg(rgb(HOVER())))
        .child(icon(name).size(px(14.)).text_color(rgb(MUTED())))
        .child(text.to_string())
}

/// True pixel art at any zoom: every sprite pixel is one solid quad whose edges are snapped to device pixels,
/// so cells never blur (gpui samples textures smoothly) and neighbours share edges (no seams, no uneven cells).
fn paint_pixels(b: Bounds<Pixels>, im: &RgbaImage, scale: f32, window: &mut Window) {
    let (iw, ih) = im.dimensions();
    if iw == 0 || ih == 0 {
        return;
    }
    let (bw, bh) = (f32::from(b.size.width), f32::from(b.size.height));
    let cell = (bw / iw as f32).min(bh / ih as f32); // object-fit: contain
    let (ox, oy) = (f32::from(b.origin.x) + (bw - cell * iw as f32) / 2., f32::from(b.origin.y) + (bh - cell * ih as f32) / 2.);
    let snap = |v: f32| (v * scale).round() / scale;
    // only what is visible: the clip of the enclosing viewer (overflow_hidden), not the whole window, so a 2400 % zoom
    // paints the few cells in the square instead of every cell that happens to fall inside the window
    let clip = window.content_mask().bounds;
    let (cx0, cy0) = (f32::from(clip.origin.x), f32::from(clip.origin.y));
    let (cx1, cy1) = (cx0 + f32::from(clip.size.width), cy0 + f32::from(clip.size.height));
    let xs = (((cx0 - ox) / cell).floor().max(0.) as u32).min(iw);
    let xe = (((cx1 - ox) / cell).ceil().max(0.) as u32).min(iw);
    for y in 0..ih {
        let (y0, y1) = (snap(oy + y as f32 * cell), snap(oy + (y + 1) as f32 * cell));
        if y1 <= cy0 || y0 >= cy1 || y1 <= y0 {
            continue;
        }
        // runs of the same colour are one quad: pixel art is mostly flat areas, so this cuts the quad count a lot
        let mut x = xs;
        while x < xe {
            let p = im.get_pixel(x, y).0;
            let mut e = x + 1;
            while e < xe && im.get_pixel(e, y).0 == p {
                e += 1;
            }
            if p[3] != 0 {
                let (x0, x1) = (snap(ox + x as f32 * cell), snap(ox + e as f32 * cell));
                if x1 > x0 {
                    let c = rgba(((p[0] as u32) << 24) | ((p[1] as u32) << 16) | ((p[2] as u32) << 8) | p[3] as u32);
                    window.paint_quad(fill(Bounds::new(point(px(x0), px(y0)), size(px(x1 - x0), px(y1 - y0))), c));
                }
            }
            x = e;
        }
    }
}

/// The live preview field, fitted to the bounds; it keeps asking for frames while it is on screen.
fn paint_field(b: Bounds<Pixels>, tex: &Arc<RenderImage>, window: &mut Window) {
    let s = tex.size(0);
    let (iw, ih) = (s.width.0 as f32, s.height.0 as f32);
    if iw > 0. && ih > 0. {
        let (bw, bh) = (f32::from(b.size.width), f32::from(b.size.height));
        let k = (bw / iw).min(bh / ih);
        let (w, h) = (iw * k, ih * k);
        let at = point(b.origin.x + px((bw - w) / 2.), b.origin.y + px((bh - h) / 2.));
        // one texel past every side, clipped: the texture's own edge samples blend with what is around it
        let image = Bounds::new(at, size(px(w), px(h)));
        let over = Bounds::new(point(at.x - px(k), at.y - px(k)), size(px(w + 2. * k), px(h + 2. * k)));
        window.with_content_mask(Some(gpui::ContentMask { bounds: image }), |window| {
            let _ = window.paint_image(over, gpui::Corners::default(), tex.clone(), 0, false);
        });
    }
    window.request_animation_frame();
}

fn paint_checker(b: Bounds<Pixels>, cell: f32, window: &mut Window) {
    window.paint_quad(fill(b, rgb(CHECK_A())));
    let (w, h) = (f32::from(b.size.width), f32::from(b.size.height));
    let (nx, ny) = ((w / cell).ceil() as i32, (h / cell).ceil() as i32);
    for y in 0..ny {
        for x in 0..nx {
            if (x + y) % 2 == 0 {
                continue;
            }
            let o = point(b.origin.x + px(x as f32 * cell), b.origin.y + px(y as f32 * cell));
            let s = size(px(cell.min(w - x as f32 * cell)), px(cell.min(h - y as f32 * cell)));
            window.paint_quad(fill(Bounds::new(o, s), rgb(CHECK_B())));
        }
    }
}

fn fmt_secs(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 60 { format!("{}:{:02}", s / 60, s % 60) } else { format!("{s} s") }
}

// ================================================================ render
impl Studio {
    /// GPU state as (dot colour, text), shown in the status bar.
    fn gpu_status(&self) -> (u32, String) {
        if self.activity.loading.is_some() {
            return (WARN(), tr!("cargando", "loading").into());
        }
        match self.phase.as_str() {
            "loading" => (WARN(), tr!("cargando", "loading").into()),
            "ready" => (BLUE(), tr!("GPU lista", "GPU ready").into()),
            "busy" => (BLUE(), tr!("generando", "generating").into()),
            _ => (MUTED_2(), tr!("GPU libre", "GPU idle").into()),
        }
    }

    /// Loading progress 0..1 from the weight bytes read (None: size unknown -> indeterminate) and the
    /// "42 % · ~6 s left" text.
    pub(super) fn load_progress(&self) -> Option<(String, Option<f32>, String)> {
        let l = self.activity.loading.as_ref()?;
        let frac = l.fraction();
        let txt = match (frac, l.eta()) {
            (Some(f), Some(e)) => tr!(
                format!("{:.0} % · quedan ~{}", f * 100., fmt_secs(Duration::from_secs_f64(e.max(1.)))),
                format!("{:.0} % · ~{} left", f * 100., fmt_secs(Duration::from_secs_f64(e.max(1.))))
            ),
            (Some(f), None) => format!("{:.0} %", f * 100.),
            (None, _) => fmt_secs(l.since.elapsed()),
        };
        Some((l.what.clone(), frac, txt))
    }

    fn titlebar(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        // caption buttons drive the window themselves (GPUI's caption hit test does not fire here);
        // glyphs from Segoe Fluent Icons, the font Windows 11 draws its own caption buttons with
        let caption = |id: &'static str, name: &'static str, hover: u32, hover_text: u32| {
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
                .child(icon(name).size(px(15.)).text_color(rgb(MUTED())).group_hover(id, move |s| s.text_color(rgb(hover_text))))
        };
        // GPUI's own caption hit test misses here; hand the drag to Windows directly
        let drag = |ev: &MouseDownEvent, window: &mut Window, cx: &mut gpui::App| {
            if ev.click_count == 2 {
                window.zoom_window();
            } else {
                begin_native_move(window);
            }
            cx.stop_propagation();
        };
        let chip = self.model_chip(cx);
        let downloading = self.downloads.progress().running;
        let minimal = self.welcome_visible();
        div()
            .h(px(42.))
            .flex_none()
            .relative()
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
                    .gap(px(10.))
                    .pl(px(14.))
                    .window_control_area(WindowControlArea::Drag)
                    .child(img(self.logo.clone()).w(px(24.)).h(px(24.)))
                    .child(div().text_size(px(14.)).font_weight(FontWeight::BOLD).text_color(rgb(TEXT())).child("Dotsmith"))
                    .on_mouse_down(MouseButton::Left, drag),
            )
            .child(div().id("drag-right").flex_1().h_full().window_control_area(WindowControlArea::Drag).on_mouse_down(MouseButton::Left, drag))
            .when(!minimal, |d| {
                d.child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(2.))
                        .pr(px(8.))
                        .child(
                            bar_btn("open-palette", "search", tr!("Buscar · Ctrl+K", "Search · Ctrl+K"), self.palette_open, MUTED())
                                .on_click(cx.listener(|this, _, w, cx| this.open_palette(w, cx))),
                        )
                        .child(bar_btn("open-tools", "wrench", tr!("Herramientas", "Tools"), self.menu_open, MUTED()).on_click(cx.listener(
                            |this, _, _, cx| {
                                this.menu_open = !this.menu_open;
                                cx.notify();
                            },
                        )))
                        .child(
                            bar_btn(
                                "open-downloads",
                                "download",
                                tr!("Descargas", "Downloads"),
                                self.downloads_open,
                                if downloading { ACCENT() } else { MUTED() },
                            )
                            .when(downloading, |d| d.child(div().absolute().top(px(6.)).right(px(7.)).w(px(6.)).h(px(6.)).rounded_full().bg(rgb(ACCENT()))))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.downloads_open = !this.downloads_open;
                                cx.notify();
                            })),
                        )
                        .child(bar_btn("open-settings", "settings", tr!("Ajustes · Ctrl+,", "Settings · Ctrl+,"), self.settings_open, MUTED()).on_click(
                            cx.listener(|this, _, _, cx| {
                                this.settings_open = !this.settings_open;
                                cx.notify();
                            }),
                        )),
                )
                .child(div().w(px(1.)).h(px(18.)).bg(rgb(BORDER_SOFT())))
            })
            .child(caption("min", "minus", HOVER(), TEXT()).on_click(|_, window, _| window.minimize_window()))
            .child(caption("max", if self.maximized { "copy" } else { "square" }, HOVER(), TEXT()).on_click(|_, window, _| window.zoom_window()))
            .child(caption("close", "x", 0xc42b1c, 0xffffff).on_click(|_, window, _| window.remove_window()))
            // the model selector sits on the window's centre line, whatever is on either side
            .when(!minimal, |d| d.child(div().absolute().top_0().left_0().size_full().flex().items_center().justify_center().child(chip)))
    }

    fn rail(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        // short windows (a 1080p laptop at 150 %) get tighter items, and the rail scrolls if even those do not fit
        let item_pad = if self.viewport.1 < 760. { 5. } else { 9. };
        div()
            .id("rail")
            .overflow_y_scroll()
            .w(px(74.))
            .flex_none()
            .flex()
            .flex_col()
            .items_center()
            .gap(px(4.))
            .pt(px(12.))
            .bg(rgb(BG()))
            .border_r_1()
            .border_color(rgb(BORDER_SOFT()))
            .children(MODES.iter().enumerate().flat_map(|(i, &m)| {
                let active = self.mode == m;
                let divider = (i == CREATE_MODES).then(|| div().w(px(36.)).h(px(1.)).my(px(6.)).bg(rgb(BORDER())).into_any_element());
                let ok = self.mode_ok(m);
                let tip = if ok {
                    m.title().to_string()
                } else {
                    let name = self.family_info().map(|f| f.label.clone()).unwrap_or_default();
                    tr!(format!("{} · {name} no lo hace", m.title()), format!("{} · {name} can't do this", m.title()))
                };
                let item = div()
                    .kb(("mode", i))
                    .w(px(60.))
                    .py(px(item_pad))
                    .flex()
                    .flex_col()
                    .items_center()
                    .gap(px(4.))
                    .rounded(px(9.))
                    .cursor_pointer()
                    .tooltip(tooltip(&tip))
                    .when(!ok, |d| d.opacity(0.4))
                    .when(active, |d| d.bg(rgb(RAISED())))
                    .hover(|s| s.bg(rgb(PANEL())))
                    .child(icon(m.glyph()).size(px(19.)).text_color(rgb(if active { ACCENT() } else { MUTED() })))
                    .child(div().text_size(px(10.5)).text_color(rgb(if active { TEXT() } else { MUTED() })).child(m.label()))
                    .on_click(cx.listener(move |this, _, w, cx| {
                        this.close_prompt_modal(&ClosePromptModal, w, cx);
                        this.mode = m;
                        cx.notify();
                    }))
                    .into_any_element();
                divider.into_iter().chain(std::iter::once(item))
            }))
    }

    fn source_panel(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let body: AnyElement = match &self.source {
            Some(s) => div()
                .flex()
                .gap(px(12.))
                .items_center()
                .child(
                    div()
                        .w(px(96.))
                        .h(px(96.))
                        .flex_none()
                        .rounded(px(7.))
                        .overflow_hidden()
                        .bg(rgb(CHECK_A()))
                        .border_1()
                        .border_color(rgb(BORDER()))
                        .child(img(s.view.clone()).size_full().object_fit(ObjectFit::Contain)),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap(px(4.))
                        .child(
                            div()
                                .text_size(px(12.))
                                .text_color(rgb(TEXT()))
                                .overflow_hidden()
                                .text_ellipsis()
                                .whitespace_nowrap()
                                .child(s.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()),
                        )
                        .child(div().text_size(px(11.)).text_color(rgb(MUTED())).child(format!("{} × {} px", s.w, s.h)))
                        .child(
                            div()
                                .flex()
                                .gap(px(6.))
                                .pt(px(4.))
                                .child(button("src-open", tr!("Cambiar…", "Change…")).on_click(cx.listener(|this, _, _, cx| this.open_source(cx))))
                                .child(button("src-clear", tr!("Quitar", "Remove")).on_click(cx.listener(|this, _, _, cx| {
                                    if let Some(s) = this.source.take() {
                                        this.trash.push(s.view);
                                    }
                                    cx.notify();
                                }))),
                        ),
                )
                .into_any_element(),
            None => div()
                .kb("drop-hint")
                .h(px(96.))
                .flex()
                .flex_col()
                .items_center()
                .justify_center()
                .gap(px(6.))
                .rounded(px(8.))
                .border_1()
                .border_dashed()
                .border_color(rgb(BORDER()))
                .bg(rgb(INPUT_BG()))
                .cursor_pointer()
                .hover(|s| s.bg(rgb(PANEL())).border_color(rgb(ACCENT_DIM())))
                .child(icon("image-plus").size(px(20.)).text_color(rgb(MUTED())))
                .child(div().text_size(px(12.5)).text_color(rgb(TEXT())).child(tr!("Elegir una imagen", "Choose an image")))
                .child(
                    div()
                        .text_size(px(11.))
                        .text_color(rgb(MUTED_2()))
                        .child(tr!("o arrástrala, o pégala con Ctrl+V", "or drop it here, or paste with Ctrl+V")),
                )
                .on_click(cx.listener(|this, _, _, cx| this.open_source(cx)))
                .into_any_element(),
        };
        div()
            .id("source")
            .rounded(px(9.))
            .border_2()
            .border_color(gpui::transparent_black())
            .drag_over::<DraggedSprite>(|s, _, _, _| s.border_color(rgb(BLUE())).bg(rgba(0xe5a95f14)))
            .on_drop(cx.listener(|this, d: &DraggedSprite, _, cx| {
                this.set_source(d.png.clone(), cx);
                cx.stop_propagation();
            }))
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                if let Some(p) = paths.paths().first() {
                    this.set_source(p.clone(), cx);
                }
            }))
            .child(body)
            .into_any_element()
    }

    fn mask_panel(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let Some(sprite) = self.source.as_ref().and_then(|s| s.sprite.clone()) else {
            return div()
                .text_size(px(12.))
                .text_color(rgb(MUTED_2()))
                .child(if self.source.is_some() {
                    tr!("Preparando el sprite…", "Preparing the sprite…")
                } else {
                    tr!("Carga primero el sprite.", "Load the sprite first.")
                })
                .into_any_element();
        };
        let mask = self.mask.clone();
        let gb = self.grid.clone();
        let count = mask.iter().filter(|&&b| b).count();
        let tools = [(Tool::Brush, tr!("Pincel", "Brush")), (Tool::Erase, tr!("Goma", "Eraser")), (Tool::Wand, tr!("Color", "Colour"))];
        let grid = div()
            .id("mask")
            .w(px(320.))
            .h(px(320.))
            .rounded(px(6.))
            .overflow_hidden()
            .border_1()
            .border_color(rgb(BORDER()))
            .cursor_crosshair()
            .on_mouse_down(MouseButton::Left, cx.listener(|this, ev: &MouseDownEvent, _, cx| this.on_grid_down(ev, false, cx)))
            .on_mouse_down(MouseButton::Right, cx.listener(|this, ev: &MouseDownEvent, _, cx| this.on_grid_down(ev, true, cx)))
            .on_mouse_move(cx.listener(|this, ev: &MouseMoveEvent, _, cx| this.on_grid_move(ev, cx)))
            .on_mouse_up(MouseButton::Left, cx.listener(|this, _: &MouseUpEvent, _, _| this.painting = None))
            .on_mouse_up(MouseButton::Right, cx.listener(|this, _: &MouseUpEvent, _, _| this.painting = None))
            .on_mouse_up_out(MouseButton::Left, cx.listener(|this, _: &MouseUpEvent, _, _| this.painting = None))
            .child(
                canvas(
                    move |b, _, _| gb.set(Some(b)),
                    move |b, _, window, _| {
                        paint_checker(b, f32::from(b.size.width) / 16., window);
                        let c = b.size.width / 32.;
                        for (i, p) in sprite.pixels().enumerate() {
                            let (x, y) = ((i % 32) as f32, (i / 32) as f32);
                            let o = point(b.origin.x + c * x, b.origin.y + c * y);
                            let p = p.0;
                            if p[3] > 0 {
                                let col = gpui::Rgba { r: p[0] as f32 / 255., g: p[1] as f32 / 255., b: p[2] as f32 / 255., a: p[3] as f32 / 255. };
                                window.paint_quad(fill(Bounds::new(o, size(c, c)), col));
                            }
                            if mask[i] {
                                window.paint_quad(fill(Bounds::new(o, size(c, c)), rgba(0xeee2cb48)).border_widths(px(1.)).border_color(rgba(0xf7eedccc)));
                            }
                        }
                    },
                )
                .size_full(),
            );
        div()
            .flex()
            .flex_col()
            .gap(px(8.))
            .child(
                div()
                    .flex()
                    .gap(px(6.))
                    .items_center()
                    .children(tools.iter().enumerate().map(|(i, (t, name))| {
                        let t = *t;
                        chip(("tool", i), name, self.tool == t).on_click(cx.listener(move |this, _, _, cx| {
                            this.tool = t;
                            cx.notify();
                        }))
                    }))
                    .child(div().w(px(8.)))
                    .children((1..=3).map(|b| {
                        chip(("brush", b as usize), &format!("{}px", b * 2 - 1), self.brush == b).on_click(cx.listener(move |this, _, _, cx| {
                            this.brush = b;
                            cx.notify();
                        }))
                    })),
            )
            .child(grid)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(
                        div()
                            .flex_1()
                            .text_size(px(11.))
                            .text_color(rgb(MUTED()))
                            .child(tr!(format!("{count} píxeles editables · clic derecho borra"), format!("{count} editable pixels · right click erases"))),
                    )
                    .child(button("m-all", tr!("Todo", "All")).on_click(cx.listener(|this, _, _, cx| {
                        if let Some(sp) = this.source.as_ref().and_then(|s| s.sprite.as_ref()) {
                            this.mask = sp.pixels().map(|p| p.0[3] > 0).collect();
                        }
                        cx.notify();
                    })))
                    .child(button("m-inv", tr!("Invertir", "Invert")).on_click(cx.listener(|this, _, _, cx| {
                        if let Some(sp) = this.source.as_ref().and_then(|s| s.sprite.as_ref()) {
                            this.mask = sp.pixels().zip(this.mask.iter()).map(|(p, &m)| p.0[3] > 0 && !m).collect();
                        }
                        cx.notify();
                    })))
                    .child(button("m-none", tr!("Nada", "None")).on_click(cx.listener(|this, _, _, cx| {
                        this.mask = vec![false; 1024];
                        cx.notify();
                    }))),
            )
            .into_any_element()
    }

    fn nudge_lora(&mut self, d: f32, cx: &mut Context<Self>) {
        self.lora_scale = ((self.lora_scale + d) * 20.).round().clamp(0., 30.) / 20.;
        self.settings.lora_scale = Some(self.lora_scale);
        let _ = self.settings.save();
        cx.notify();
    }

    fn stepper(
        &mut self,
        id: &'static str,
        name: &str,
        value: String,
        hint: &str,
        delta: impl Fn(&mut Form, f32) + Clone + 'static,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let d1 = delta.clone();
        div()
            .flex()
            .items_center()
            .gap(px(8.))
            .py(px(4.))
            .child(
                div()
                    .id((id, 9usize))
                    .flex_1()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .when(!hint.is_empty(), |d| d.tooltip(tooltip(hint)))
                    .child(div().text_size(px(12.5)).text_color(rgb(TEXT())).child(name.to_string()))
                    .when(!hint.is_empty(), |d| d.child(info_mark())),
            )
            .child(icon_btn((id, 0usize), "minus").on_click(cx.listener(move |this, _, _, cx| {
                d1(this.form_mut(), -1.);
                cx.notify();
            })))
            .child(div().w(px(52.)).flex().justify_center().font_family(MONO).text_size(px(12.5)).text_color(rgb(ACCENT())).child(value))
            .child(icon_btn((id, 1usize), "plus").on_click(cx.listener(move |this, _, _, cx| {
                delta(this.form_mut(), 1.);
                cx.notify();
            })))
    }

    // ------------------------------------------------------------ pose / silhouette
    fn pose_image(&self, sheet: bool) -> RgbaImage {
        if sheet { rig::sheet(&self.pose) } else { rig::render(&self.pose) }
    }

    /// Saves the pose (frame or 4x2 sheet) next to the other results; `as_source` sends it to Variantes as the origin.
    fn save_pose(&mut self, as_source: bool, cx: &mut Context<Self>) {
        let dir = self.model.out_dir.clone();
        if dir.as_os_str().is_empty() {
            self.say(tr!("Espera a que arranque el motor.", "Wait for the engine to start."), Level::Error);
            cx.notify();
            return;
        }
        let sheet = self.pose_sheet && !as_source;
        let img = self.pose_image(sheet);
        let p = self.pose;
        let tag = if sheet { "sheet".to_string() } else { format!("f{}", p.frame) };
        let name = format!("{}_pose_{}_{}_{}", crate::jobs::chrono_stamp(), p.kind.key(), p.dir.key(), tag);
        let png = dir.join(format!("{name}.png"));
        let json = json!({"mode": "pose", "prompt": "", "rig": p.kind.key(), "dir": p.dir.key(), "anim": p.anim.key(), "frame": p.frame,
            "sheet": sheet, "projection": if p.oblique { "tibia oblique" } else { "orthographic" }, "lean": p.lean, "pitch": p.pitch,
            "silhouette": p.silhouette, "width": img.width(), "height": img.height()});
        let res = std::fs::create_dir_all(&dir)
            .map_err(|e| e.to_string())
            .and_then(|_| img.save(&png).map_err(|e| e.to_string()))
            .and_then(|_| std::fs::write(png.with_extension("json"), serde_json::to_string_pretty(&json).unwrap()).map_err(|e| e.to_string()));
        if let Err(e) = res {
            self.say(tr!(format!("No se pudo guardar la pose: {e}"), format!("Could not save the pose: {e}")), Level::Error);
            cx.notify();
            return;
        }
        if let Some(item) = pixels::Item::load(png.clone(), json) {
            self.history.insert(0, item);
        }
        if as_source {
            // a flat colour blockout needs a strong redraw to become a sprite
            self.forms[Mode::Variants as usize].strength = 0.8;
            self.mode = Mode::Variants;
            self.pose_playing = false;
            self.set_source(png.clone(), cx);
            self.say(
                tr!("Pose enviada a Variantes. Describe la criatura y genera.", "Pose sent to Variants. Describe the creature and generate."),
                Level::Info,
            );
        } else {
            self.select(Some(png.clone()));
            let name = png.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            self.say(tr!(format!("Pose guardada: {name}"), format!("Pose saved: {name}")), Level::Info);
        }
        cx.notify();
    }

    fn pose_controls(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let p = self.pose;
        let row = |items: Vec<AnyElement>| div().flex().flex_wrap().gap(px(6.)).children(items);
        let mut col = div()
            .flex()
            .flex_col()
            .px(px(18.))
            .pb(px(18.))
            .child(div().pt(px(18.)).text_size(px(17.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(Mode::Pose.title()));
        col = col.child(section("Rig")).child(row(rig::KINDS
            .iter()
            .enumerate()
            .map(|(i, (k, es, en))| {
                let k = *k;
                chip(("kind", i), tr!(*es, *en), p.kind == k)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.pose.kind = k;
                        cx.notify();
                    }))
                    .into_any_element()
            })
            .collect()));
        col = col.child(section(tr!("Dirección", "Direction"))).child(row(rig::DIRS
            .iter()
            .enumerate()
            .map(|(i, (d, es, en))| {
                let d = *d;
                chip(("dir", i), tr!(*es, *en), p.dir == d)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.pose.dir = d;
                        cx.notify();
                    }))
                    .into_any_element()
            })
            .collect()));
        col = col
            .child(section(tr!("Animación", "Animation")))
            .child(row(rig::ANIMS
                .iter()
                .enumerate()
                .map(|(i, (a, es, en))| {
                    let a = *a;
                    chip(("anim", i), tr!(*es, *en), p.anim == a)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.pose.anim = a;
                            cx.notify();
                        }))
                        .into_any_element()
                })
                .collect()))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .pt(px(8.))
                    .child(icon_btn("play", if self.pose_playing { "pause" } else { "play" }).on_click(cx.listener(|this, _, _, cx| {
                        this.pose_playing = !this.pose_playing;
                        cx.notify();
                    })))
                    .child(div().flex_1().text_size(px(12.5)).text_color(rgb(TEXT())).child(format!("Frame {} / {}", p.frame + 1, rig::FRAMES)))
                    .child(icon_btn(("frame", 0usize), "minus").on_click(cx.listener(|this, _, _, cx| {
                        this.pose_playing = false;
                        this.pose.frame = (this.pose.frame + rig::FRAMES - 1) % rig::FRAMES;
                        cx.notify();
                    })))
                    .child(icon_btn(("frame", 1usize), "plus").on_click(cx.listener(|this, _, _, cx| {
                        this.pose_playing = false;
                        this.pose.frame = (this.pose.frame + 1) % rig::FRAMES;
                        cx.notify();
                    }))),
            );
        let angle_row: AnyElement = if p.oblique {
            self.pose_stepper("lean", tr!("Inclinación", "Lean"), format!("{:.2}", p.lean), |q, d| q.lean = (q.lean + d * 0.05).clamp(0.2, 1.2), cx)
                .into_any_element()
        } else {
            self.pose_stepper(
                "pitch",
                tr!("Ángulo de cámara", "Camera angle"),
                format!("{:.0}°", p.pitch),
                |q, d| q.pitch = (q.pitch + d * 5.).clamp(0., 90.),
                cx,
            )
            .into_any_element()
        };
        col = col
            .child(section(tr!("Proyección", "Projection")))
            .child(row(vec![
                chip("proj-obl", tr!("Oblicua clásica", "Classic oblique"), p.oblique)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.pose.oblique = true;
                        cx.notify();
                    }))
                    .into_any_element(),
                chip("proj-ort", tr!("Cámara 3/4", "3/4 camera"), !p.oblique)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.pose.oblique = false;
                        cx.notify();
                    }))
                    .into_any_element(),
            ]))
            .child(angle_row);
        col = col
            .child(section(tr!("Formato", "Format")))
            .child(row([32u32, 48, 64]
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let s = *s;
                    chip(("psize", i), &format!("{s}"), p.size == s)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.pose.size = s;
                            cx.notify();
                        }))
                        .into_any_element()
                })
                .collect()))
            .child(
                div()
                    .pt(px(8.))
                    .child(toggle("silhouette", tr!("Silueta sólida", "Solid silhouette"), "", p.silhouette).on_click(cx.listener(|this, _, _, cx| {
                        this.pose.silhouette = !this.pose.silhouette;
                        cx.notify();
                    })))
                    .child(toggle("sheet", tr!("Guardar hoja de 8 frames", "Save an 8-frame sheet"), "", self.pose_sheet).on_click(cx.listener(
                        |this, _, _, cx| {
                            this.pose_sheet = !this.pose_sheet;
                            cx.notify();
                        },
                    ))),
            );
        let footer = div()
            .flex_none()
            .flex()
            .gap(px(8.))
            .p(px(14.))
            .border_t_1()
            .border_color(rgb(BORDER_SOFT()))
            .bg(rgb(PANEL()))
            .child(
                div()
                    .kb("go")
                    .flex_1()
                    .h(px(44.))
                    .rounded(px(9.))
                    .flex()
                    .items_center()
                    .justify_center()
                    .gap(px(8.))
                    .bg(rgb(ACCENT()))
                    .text_color(rgb(ACCENT_INK()))
                    .font_weight(FontWeight::BOLD)
                    .text_size(px(14.))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(ACCENT_HOVER())))
                    .child(tr!("Usar como origen", "Use as source"))
                    .on_click(cx.listener(|this, _, _, cx| this.save_pose(true, cx))),
            )
            .child(
                div()
                    .id("save-pose")
                    .w(px(110.))
                    .h(px(44.))
                    .rounded(px(9.))
                    .flex()
                    .items_center()
                    .justify_center()
                    .border_1()
                    .border_color(rgb(BORDER()))
                    .text_size(px(13.))
                    .text_color(rgb(TEXT()))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(HOVER())))
                    .child(tr!("Guardar", "Save"))
                    .on_click(cx.listener(|this, _, _, cx| this.save_pose(false, cx))),
            );
        div()
            .id("controls-col")
            .w(px(380.))
            .flex_none()
            .flex()
            .flex_col()
            .bg(rgb(PANEL()))
            .border_r_1()
            .border_color(rgb(BORDER_SOFT()))
            .child(div().id("controls").flex_1().min_h_0().overflow_y_scroll().child(col))
            .child(footer)
            .into_any_element()
    }

    fn pose_stepper(
        &mut self,
        id: &'static str,
        name: &str,
        value: String,
        delta: impl Fn(&mut rig::Pose, f32) + Clone + 'static,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let d1 = delta.clone();
        div()
            .flex()
            .items_center()
            .gap(px(8.))
            .py(px(6.))
            .child(div().flex_1().text_size(px(12.5)).text_color(rgb(TEXT())).child(name.to_string()))
            .child(icon_btn((id, 0usize), "minus").on_click(cx.listener(move |this, _, _, cx| {
                d1(&mut this.pose, -1.);
                cx.notify();
            })))
            .child(div().w(px(52.)).flex().justify_center().font_family(MONO).text_size(px(12.5)).text_color(rgb(ACCENT())).child(value))
            .child(icon_btn((id, 1usize), "plus").on_click(cx.listener(move |this, _, _, cx| {
                delta(&mut this.pose, 1.);
                cx.notify();
            })))
    }

    fn controls(&mut self, cx: &mut Context<Self>) -> AnyElement {
        if self.mode == Mode::Pose {
            return self.pose_controls(cx);
        }
        let mode = self.mode;
        let f = self.form();
        let prompt = f.prompt.clone();
        let seed = f.seed.clone();
        let (camera, size_ix, opaque, center, mirror) = (f.camera, f.size, f.opaque, f.center, f.mirror);
        let (strength, variants, guidance) = (f.strength, f.variants, f.guidance);
        let (steps, steps_default) = (self.steps_of(f), self.default_steps());
        let turbo = self.model.turbo;
        let lora_scale = self.lora_scale;
        let has_lora = !self.model.lora.is_empty();
        let mut col = div()
            .flex()
            .flex_col()
            .px(px(18.))
            .pb(px(18.))
            .child(div().pt(px(18.)).text_size(px(17.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(mode.title()));
        if mode.uses_source() {
            col = col.child(section(tr!("Origen", "Source"))).child(self.source_panel(cx));
        }
        if mode == Mode::Regions && self.source.is_some() {
            col = col.child(section(tr!("Zona editable", "Editable area"))).child(self.mask_panel(cx));
        }
        let field: AnyElement = if self.prompt_modal {
            div()
                .id("prompt-away")
                .h(px(84.))
                .flex()
                .items_center()
                .justify_center()
                .rounded(px(7.))
                .bg(rgb(INPUT_BG()))
                .border_1()
                .border_dashed()
                .border_color(rgb(BORDER()))
                .text_size(px(12.))
                .text_color(rgb(MUTED()))
                .cursor_pointer()
                .child(tr!("Editando en la ventana ampliada…", "Editing in the large editor…"))
                .on_click(cx.listener(|this, _, w, cx| this.open_prompt_modal(w, cx)))
                .into_any_element()
        } else {
            div()
                .id("prompt-drop")
                .relative()
                .rounded(px(9.))
                .border_2()
                .border_color(gpui::transparent_black())
                .drag_over::<DraggedSprite>(|s, _, _, _| s.border_color(rgb(BLUE())))
                .on_drop(cx.listener(|this, d: &DraggedSprite, _, cx| {
                    let text = this.history.iter().find(|i| i.png == d.png).and_then(|i| i.json["prompt"].as_str().map(String::from));
                    if let Some(t) = text {
                        this.form().prompt.update(cx, |p, cx| p.set_text(&t, cx));
                        this.say(tr!("Descripción copiada del resultado.", "Description copied from the result."), Level::Info);
                    }
                    cx.stop_propagation();
                    cx.notify();
                }))
                .child(prompt)
                .child(
                    div()
                        .kb("expand")
                        .absolute()
                        .top(px(5.))
                        .right(px(5.))
                        .w(px(24.))
                        .h(px(24.))
                        .rounded(px(6.))
                        .flex()
                        .items_center()
                        .justify_center()
                        .bg(rgb(PANEL()))
                        .border_1()
                        .border_color(rgb(BORDER_SOFT()))
                        .cursor_pointer()
                        .hover(|s| s.bg(rgb(HOVER())).text_color(rgb(TEXT())))
                        .child(icon("maximize-2").size(px(13.)).text_color(rgb(MUTED())))
                        .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                        .on_click(cx.listener(|this, _, w, cx| this.open_prompt_modal(w, cx))),
                )
                .into_any_element()
        };
        col = col.child(section(if mode == Mode::Pack { tr!("Dirección del pack", "Pack direction") } else { tr!("Descripción", "Description") })).child(field);
        if mode == Mode::Creature {
            col = col.child(self.creature_options(cx));
        }
        if mode == Mode::Walls {
            let room = self.form().room;
            col = col.child(section(tr!("Muros", "Walls"))).child(div().pt(px(4.)).child(
                toggle("room", tr!("Montar habitación 3×3", "Build a 3×3 room"), "", room).on_click(cx.listener(|this, _, _, cx| {
                    let f = this.form_mut();
                    f.room = !f.room;
                    cx.notify();
                })),
            ));
        }
        // format
        let sizes: Vec<(usize, String)> = match mode {
            Mode::Single => SINGLE_SIZES
                .iter()
                .enumerate()
                .filter(|(_, s)| self.size_ok(s.1, s.2) && (s.1 >= 32 || self.family_info().is_some_and(|f| f.sizes.contains(&s.1))))
                .map(|(i, s)| (i, s.0.to_string()))
                .collect(),
            Mode::Pack => PACKS.iter().enumerate().map(|(i, s)| (i, s.0.to_string())).collect(),
            Mode::Variants | Mode::Angle => OUT_SIZES.iter().enumerate().filter(|(_, s)| self.size_ok(**s, **s)).map(|(i, s)| (i, format!("{s}"))).collect(),
            _ => vec![],
        };
        let own_sizes = self.family_info().is_some_and(|f| !f.sizes.is_empty());
        let plain = self.plain_family();
        if !sizes.is_empty() {
            let experimental = |i: usize| !own_sizes && mode == Mode::Single && SINGLE_SIZES[i].1.max(SINGLE_SIZES[i].2) > 32;
            col = col.child(section(tr!("Formato", "Format"))).child(div().flex().flex_wrap().gap(px(6.)).children(sizes.into_iter().map(|(i, s)| {
                chip(("size", i), &s, size_ix == i)
                    .when(experimental(i), |d| d.tooltip(tooltip(tr!("Experimental: por encima de 32 px", "Experimental: above 32 px"))))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.form_mut().size = i;
                        cx.notify();
                    }))
            })));
        }
        if mode == Mode::Angle {
            col = col.child(section(tr!("Ángulo", "Angle"))).child(self.gizmo_panel(cx));
        }
        if !plain && !matches!(mode, Mode::Photo | Mode::Angle | Mode::Creature | Mode::Walls) {
            col = col.child(div().pt(px(12.)).pb(px(6.)).child(label(tr!("Ángulo de cámara", "Camera angle")))).child(div().flex().gap(px(6.)).children(
                (0..CAMERAS.len()).map(|i| {
                    chip(("cam", i), camera_label(i), camera == i)
                        .when(i == 2, |d| d.tooltip(tooltip(tr!("Para cofres, carteles y muebles", "For chests, signs and furniture"))))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.form_mut().camera = i;
                            cx.notify();
                        }))
                }),
            ));
        }
        if matches!(mode, Mode::Single | Mode::Pack) {
            let opaque_ok = self.family_info().is_none_or(|f| f.opaque);
            col = col.child(
                div()
                    .pt(px(8.))
                    .when(opaque_ok, |d| {
                        d.child(toggle("opaque", tr!("Fondo opaco", "Opaque background"), "", opaque).on_click(cx.listener(|this, _, _, cx| {
                            let f = this.form_mut();
                            f.opaque = !f.opaque;
                            cx.notify();
                        })))
                    })
                    .child(
                        toggle(
                            "center",
                            tr!("Centrar en la celda", "Centre in the cell"),
                            tr!("Apagado: apoyado abajo, como un objeto del mapa", "Off: resting on the bottom, like a map object"),
                            center,
                        )
                        .on_click(cx.listener(|this, _, _, cx| {
                            let f = this.form_mut();
                            f.center = !f.center;
                            cx.notify();
                        })),
                    )
                    .when(mode == Mode::Single, |d| {
                        d.child(toggle("mirror", tr!("Simetría", "Symmetry"), "", mirror).on_click(cx.listener(|this, _, _, cx| {
                            let f = this.form_mut();
                            f.mirror = !f.mirror;
                            cx.notify();
                        })))
                    }),
            );
        }
        let restyle = mode == Mode::Photo && self.form().restyle;
        if mode == Mode::Photo {
            let (opaque, size) = (self.form().opaque, self.form().size.min(4));
            col = col
                .child(section(tr!("Tamaño", "Size")))
                .child(div().flex().flex_wrap().gap(px(6.)).children(DIRECT_SIZES.iter().enumerate().map(|(i, s)| {
                    chip(("dsize", i), &format!("{s} px"), size == i).on_click(cx.listener(move |this, _, _, cx| {
                        this.form_mut().size = i;
                        cx.notify();
                    }))
                })))
                .child(
                    div()
                        .pt(px(8.))
                        .child(
                            toggle("restyle", tr!("Redibujar con IA", "Redraw with AI"), tr!("Apagado: solo pixelado", "Off: pixelation only"), restyle)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    let f = this.form_mut();
                                    f.restyle = !f.restyle;
                                    cx.notify();
                                })),
                        )
                        .child(
                            toggle(
                                "opaque",
                                tr!("Fondo opaco", "Opaque background"),
                                tr!(
                                    "Apagado: se quita un fondo liso y se recorta al objeto",
                                    "Off: a flat background is removed and the image cropped to the object"
                                ),
                                opaque,
                            )
                            .on_click(cx.listener(|this, _, _, cx| {
                                let f = this.form_mut();
                                f.opaque = !f.opaque;
                                cx.notify();
                            })),
                        ),
                );
        }
        if mode.uses_source() && (mode != Mode::Photo || restyle) {
            let sname = match mode {
                Mode::Regions => tr!("Intensidad en la zona", "Strength in the area"),
                Mode::Photo => tr!("Estilización", "Stylisation"),
                _ => tr!("Intensidad del cambio", "Change strength"),
            };
            let shint = if mode == Mode::Photo {
                if strength >= crate::jobs::MIX_STRENGTH {
                    tr!("Mezcla: el prompt sale de la silueta", "Mix: the prompt goes past the outline")
                } else {
                    tr!("Bajo = fiel · desde 0,60 mezcla", "Low = faithful · from 0.60 mixes")
                }
            } else {
                ""
            };
            let (lo, hi) = if mode == Mode::Photo { (0.3, 0.95) } else { (0.1, 0.9) };
            col = col
                .child(section(tr!("Transformación", "Transformation")))
                .child(self.stepper("strength", sname, format!("{strength:.2}"), shint, move |f, d| f.strength = (f.strength + d * 0.05).clamp(lo, hi), cx))
                .child(self.stepper(
                    "variants",
                    tr!("Variantes", "Variants"),
                    format!("{variants}"),
                    "",
                    |f, d| f.variants = (f.variants as i32 + d as i32).clamp(1, 9) as u32,
                    cx,
                ));
        }
        let seed_text = seed.read(cx).text().trim().to_string();
        let mut summary = format!(
            "{} · {steps} {}",
            if seed_text.is_empty() { tr!("semilla aleatoria", "random seed").to_string() } else { format!("{} {seed_text}", tr!("semilla", "seed")) },
            tr!("pasos", "steps")
        );
        if has_lora && (lora_scale - 1.).abs() > 0.001 {
            summary.push_str(&format!(" · LoRA {lora_scale:.2}"));
        }
        let open = self.advanced_open;
        col = col.child(
            div()
                .kb("advanced")
                .flex()
                .items_center()
                .gap(px(6.))
                .pt(px(18.))
                .pb(px(6.))
                .cursor_pointer()
                .on_click(cx.listener(|this, _, _, cx| {
                    this.advanced_open = !this.advanced_open;
                    cx.notify();
                }))
                .child(icon(if open { "chevron-down" } else { "chevron-right" }).size(px(14.)).text_color(rgb(MUTED())))
                .child(div().text_size(px(12.5)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(tr!("Avanzado", "Advanced")))
                .child(div().flex_1().h(px(1.)).mx(px(4.)).bg(rgb(BORDER_SOFT())))
                .when(!open, |d| d.child(div().font_family(MONO).text_size(px(10.5)).text_color(rgb(MUTED_2())).child(summary))),
        );
        if open {
            col = col
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .py(px(4.))
                        .child(div().flex_1().text_size(px(12.5)).text_color(rgb(TEXT())).child(tr!("Semilla", "Seed")))
                        .child(div().w(px(130.)).child(seed.clone()))
                        .child(
                            icon_btn("dice", "history")
                                .tooltip(tooltip(tr!("Usar la semilla del resultado seleccionado", "Use the selected result's seed")))
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    let s = this.selected_item().and_then(|i| i.json["seed"].as_i64()).map(|v| v.to_string()).unwrap_or_default();
                                    this.form().seed.update(cx, |t, cx| t.set_text(&s, cx));
                                })),
                        ),
                )
                .child(self.stepper(
                    "steps",
                    tr!("Pasos", "Steps"),
                    format!("{steps}"),
                    &tr!(
                        format!("Más pasos: más detalle y más tiempo. Por defecto {steps_default}."),
                        format!("More steps: more detail, more time. Default {steps_default}.")
                    ),
                    move |f, d| {
                        let v = Some((steps as i32 + d as i32).clamp(1, 60) as u32);
                        if turbo { f.turbo_steps = v } else { f.steps = v }
                    },
                    cx,
                ));
            if has_lora {
                col = col.child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .py(px(4.))
                        .child(
                            div()
                                .id("lora-scale-name")
                                .flex_1()
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .tooltip(tooltip(tr!(
                                    "Cuánto pesa el estilo del LoRA. 1 = como se entrenó, 0 = el modelo base solo.",
                                    "How much the LoRA's style weighs. 1 = as trained, 0 = the bare base model."
                                )))
                                .child(div().text_size(px(12.5)).text_color(rgb(TEXT())).child(tr!("Fuerza del LoRA", "LoRA strength")))
                                .child(info_mark()),
                        )
                        .child(icon_btn("lora-minus", "minus").on_click(cx.listener(|this, _, _, cx| this.nudge_lora(-0.05, cx))))
                        .child(
                            div()
                                .w(px(52.))
                                .flex()
                                .justify_center()
                                .font_family(MONO)
                                .text_size(px(12.5))
                                .text_color(rgb(ACCENT()))
                                .child(format!("{lora_scale:.2}")),
                        )
                        .child(icon_btn("lora-plus", "plus").on_click(cx.listener(|this, _, _, cx| this.nudge_lora(0.05, cx)))),
                );
            }
            if mode.uses_source() && (mode != Mode::Photo || restyle) {
                col = col.child(self.stepper(
                    "guidance",
                    tr!("Fuerza del prompt", "Prompt strength"),
                    format!("{guidance:.1}"),
                    tr!("Cuánto sigue la descripción (CFG)", "How closely it follows the description (CFG)"),
                    |f, d| f.guidance = (f.guidance + d * 0.5).clamp(1., 6.),
                    cx,
                ));
            }
        }
        let busy = self.running || self.pending;
        let running = self.running;

        let (step, total, index, count) = self.progress;
        let footer = div().flex_none().flex().p(px(14.)).border_t_1().border_color(rgb(BORDER_SOFT())).bg(rgb(PANEL())).child(
            div()
                .kb("go")
                .relative()
                .overflow_hidden()
                .flex_1()
                .h(px(44.))
                .rounded(px(9.))
                .flex()
                .items_center()
                .justify_center()
                .gap(px(8.))
                .font_weight(FontWeight::BOLD)
                .text_size(px(14.))
                .cursor_pointer()
                .when(!busy, |d| d.bg(rgb(ACCENT())).text_color(rgb(ACCENT_INK())).hover(|s| s.bg(rgb(ACCENT_HOVER()))))
                .when(busy, |d| d.bg(rgb(RAISED())).border_1().border_color(rgb(BORDER())).text_color(rgb(TEXT())).hover(|s| s.border_color(rgb(DANGER()))))
                .when(running && total > 0, |d| {
                    let frac = ((index as f32 + step as f32 / total as f32) / count.max(1) as f32).clamp(0., 1.);
                    d.child(div().absolute().left_0().top_0().bottom_0().w(gpui::relative(frac)).bg(rgba(0xeee2cb1f)))
                })
                .when(busy, |d| {
                    d.child(crate::icons::spinner("go-spin", 15., MUTED()))
                        .child(if running { tr!("Detener", "Stop") } else { tr!("Empezando…", "Starting…") })
                        .when(running && total > 0, |d| {
                            d.child(div().font_family(MONO).text_size(px(11.)).font_weight(FontWeight::NORMAL).text_color(rgb(MUTED())).child(if count > 1 {
                                format!("{}/{count} · {step}/{total}", index + 1)
                            } else {
                                format!("{step}/{total}")
                            }))
                        })
                        .child(div().text_size(px(10.5)).font_weight(FontWeight::NORMAL).opacity(0.6).child("Esc"))
                        .on_click(cx.listener(|this, _, w, cx| this.stop(&Stop, w, cx)))
                })
                .when(!busy, |d| {
                    d.child(match mode {
                        Mode::Single => tr!("Generar sprite", "Generate sprite"),
                        Mode::Pack => tr!("Generar pack", "Generate pack"),
                        Mode::Variants => tr!("Generar variantes", "Generate variants"),
                        Mode::Regions => tr!("Editar zona", "Edit area"),
                        Mode::Photo => tr!("Convertir a pixel art", "Convert to pixel art"),
                        Mode::Angle => tr!("Redibujar en este ángulo", "Redraw at this angle"),
                        Mode::Pose => tr!("Guardar pose", "Save pose"),
                        Mode::Creature => tr!("Generar personaje", "Generate character"),
                        Mode::Walls => tr!("Generar muros", "Generate walls"),
                    })
                    .child(div().text_size(px(10.5)).font_weight(FontWeight::NORMAL).opacity(0.6).child("Ctrl+↵"))
                    .on_click(cx.listener(|this, _, _, cx| this.start(cx)))
                }),
        );
        div()
            .id("controls-col")
            .w(px(380.))
            .flex_none()
            .flex()
            .flex_col()
            .bg(rgb(PANEL()))
            .border_r_1()
            .border_color(rgb(BORDER_SOFT()))
            .drag_over::<DraggedSprite>(|s, _, _, _| s.bg(rgb(RAISED())))
            .on_drop(cx.listener(|this, d: &DraggedSprite, _, cx| {
                if !this.mode.uses_source() {
                    this.mode = Mode::Variants;
                }
                this.set_source(d.png.clone(), cx);
            }))
            .child(div().id("controls").flex_1().min_h_0().overflow_y_scroll().child(col))
            .child(footer)
            .into_any_element()
    }

    fn stage(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let (step, total, index, count) = self.progress;
        let running = self.running || self.pending;
        // while generating: the live colour field; otherwise the real pixels, painted crisp
        let live_tex: Option<Arc<RenderImage>> = if running {
            let t = self.started.map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.);
            let (step, total, index, count) = self.progress;
            let settled = if total > 0 && count > 0 { (index as f32 + step as f32 / total as f32) / count as f32 } else { 0. };
            let (tex, old) = self.aurora.frame(t, settled);
            self.trash.extend(old);
            tex
        } else {
            None
        };
        let image: Option<Arc<RgbaImage>> = if running {
            None
        } else if self.mode == Mode::Pose {
            Some(Arc::new(self.pose_image(false)))
        } else {
            self.big.clone()
        };
        let frac = if total > 0 && count > 0 { (index as f32 + step as f32 / total as f32) / count as f32 } else { 0. };
        let header_right = if running {
            let elapsed = self.started.map(|s| fmt_secs(s.elapsed())).unwrap_or_default();
            if self.activity.loading.is_some() || self.phase == "loading" || total == 0 {
                format!(
                    "{} · {}",
                    if self.activity.loading.is_some() || self.phase == "loading" {
                        tr!("Cargando modelo", "Loading model")
                    } else {
                        tr!("Preparando", "Preparing")
                    },
                    elapsed
                )
            } else if count > 1 {
                tr!(
                    format!("Muestra {}/{} · paso {}/{} · {}", index + 1, count, step, total, elapsed),
                    format!("Sample {}/{} · step {}/{} · {}", index + 1, count, step, total, elapsed)
                )
            } else {
                tr!(format!("Paso {}/{} · {}", step, total, elapsed), format!("Step {}/{} · {}", step, total, elapsed))
            }
        } else if self.mode == Mode::Pose {
            String::new()
        } else {
            self.selected.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
        };
        let empty: AnyElement = if running {
            let load = self.load_progress();
            div()
                .flex()
                .flex_col()
                .items_center()
                .gap(px(10.))
                .child(crate::icons::spinner("stage-spin", 28., ACCENT_DIM()))
                .child(div().text_size(px(13.)).text_color(rgb(MUTED())).child(match &load {
                    Some((what, _, _)) => tr!(format!("Cargando {what} en la GPU"), format!("Loading {what} on the GPU")),
                    None if self.phase == "loading" || self.phase == "off" => tr!("Cargando el modelo en la GPU", "Loading the model on the GPU").into(),
                    None => tr!("Empezando…", "Starting…").into(),
                }))
                .children(load.map(|(_, frac, txt)| {
                    div()
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap(px(6.))
                        .child(model_picker::progress_bar(frac, 220.))
                        .child(div().text_size(px(11.5)).text_color(rgb(MUTED_2())).child(txt))
                }))
                .into_any_element()
        } else {
            div()
                .flex()
                .flex_col()
                .items_center()
                .gap(px(8.))
                .child(icon("image").size(px(28.)).text_color(rgb(MUTED_2())))
                .child(
                    div()
                        .text_size(px(13.))
                        .text_color(rgb(MUTED()))
                        .child(tr!("Describe lo que quieres y pulsa Ctrl + Intro.", "Describe what you want and press Ctrl + Enter.")),
                )
                .into_any_element()
        };
        let area = self.view_area.clone();
        let sq_cell = self.square.clone();
        let side = {
            let (vw, vh) = self.viewport;
            let strip = if self.job_results.len() > 1 { 76. } else { 0. };
            let details = if !running && self.selected.is_some() { DETAILS_H + 16. } else { 0. };
            let avail_w = vw - 74. - 380. - 292. - 32. - 2.;
            let avail_h = vh - 42. - 28. - 44. - 32. - strip - details - 2.;
            avail_w.min(avail_h).max(120.)
        };
        // the image fills 93 % of the square at 100 %
        let img_frac = 0.93 * self.zoom; // the image fills 93 % of the square at 100 %
        let (pan_x, pan_y) = (f32::from(self.pan.x), f32::from(self.pan.y));
        let zoomed = (self.zoom - 1.0).abs() > 0.001 || pan_x.abs() > 0.5 || pan_y.abs() > 0.5;
        let scale = self.scale_factor;
        let creature = if running {
            None
        } else {
            self.selected_item().and_then(|i| creature_mode::SheetInfo::of(&i.json).map(|s| (i.png.clone(), Arc::new(i.sprite.clone()), s)))
        };
        let player = creature.map(|(png, sheet, info)| self.creature_player(png, sheet, info, scale, cx));
        let has_player = player.is_some();
        let thumbs = (!running && !has_player && self.mode != Mode::Pose).then(|| self.small.clone()).flatten();
        let square = div()
            .id("viewer")
            .relative()
            .w(px(side))
            .h(px(side))
            .max_w_full()
            .max_h_full()
            .flex_none()
            .rounded(px(10.))
            .overflow_hidden()
            .border_1()
            .border_color(rgb(BORDER_SOFT()))
            .cursor(if self.view_drag.is_some() { gpui::CursorStyle::ClosedHand } else { gpui::CursorStyle::OpenHand })
            .on_scroll_wheel(cx.listener(|this, ev: &gpui::ScrollWheelEvent, _, cx| {
                let Some(b) = this.square.get() else { return };
                let dy = f32::from(ev.delta.pixel_delta(px(20.)).y);
                let z0 = this.zoom;
                // below 100 % shrinks the image inside the square (down to a tenth), above zooms in
                let z1 = (z0 * 1.0018f32.powf(dy)).clamp(0.1, 24.0);
                if (z1 - z0).abs() < 1e-4 {
                    return;
                }
                // keep the point under the cursor in place
                let c = b.center();
                let rel = point(ev.position.x - c.x, ev.position.y - c.y);
                let k = z1 / z0;
                this.pan = point(rel.x - (rel.x - this.pan.x) * k, rel.y - (rel.y - this.pan.y) * k);
                this.zoom = z1;
                cx.notify();
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, ev: &MouseDownEvent, _, cx| {
                    if ev.click_count == 2 {
                        this.zoom = 1.0;
                        this.pan = Point::default();
                        this.view_drag = None;
                    } else {
                        this.view_drag = Some((ev.position, this.pan));
                    }
                    cx.notify();
                }),
            )
            .on_mouse_move(cx.listener(|this, ev: &MouseMoveEvent, _, cx| {
                if let Some((start, p0)) = this.view_drag {
                    if ev.pressed_button != Some(MouseButton::Left) {
                        this.view_drag = None;
                    } else {
                        this.pan = point(p0.x + ev.position.x - start.x, p0.y + ev.position.y - start.y);
                    }
                    cx.notify();
                }
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    this.view_drag = None;
                    cx.notify();
                }),
            )
            .on_mouse_up_out(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    this.view_drag = None;
                    cx.notify();
                }),
            )
            .child(canvas(move |b, _, _| sq_cell.set(Some(b)), |b, _, window, _| paint_checker(b, 16., window)).absolute().size_full())
            .children(player)
            .when(!has_player, |d| {
                d.child(match (image, live_tex) {
                    (None, Some(tex)) => div()
                        .absolute()
                        .left(gpui::relative((1. - img_frac) / 2.))
                        .top(gpui::relative((1. - img_frac) / 2.))
                        .w(gpui::relative(img_frac))
                        .h(gpui::relative(img_frac))
                        .child(
                            div()
                                .absolute()
                                .left(px(pan_x))
                                .top(px(pan_y))
                                .size_full()
                                .child(canvas(|_, _, _| (), move |b, _, window, _| paint_field(b, &tex, window)).size_full()),
                        )
                        .into_any_element(),
                    // sized in fractions of the square so it is right on every frame, panned in pixels inside
                    (Some(im), _) => div()
                        .absolute()
                        .left(gpui::relative((1. - img_frac) / 2.))
                        .top(gpui::relative((1. - img_frac) / 2.))
                        .w(gpui::relative(img_frac))
                        .h(gpui::relative(img_frac))
                        .child(
                            div()
                                .absolute()
                                .left(px(pan_x))
                                .top(px(pan_y))
                                .size_full()
                                .child(canvas(|_, _, _| (), move |b, _, window, _| paint_pixels(b, &im, scale, window)).size_full()),
                        )
                        .into_any_element(),
                    (None, None) => div().absolute().size_full().flex().items_center().justify_center().child(empty).into_any_element(),
                })
            })
            .when(running && total > 0, |d| d.child(div().absolute().bottom_0().left_0().h(px(3.)).w(gpui::relative(frac.clamp(0., 1.))).bg(rgb(ACCENT()))))
            .when(running && self.aurora.is_live(), |d| {
                d.child(
                    div()
                        .absolute()
                        .top(px(12.))
                        .left(px(12.))
                        .px(px(8.))
                        .py(px(3.))
                        .rounded(px(5.))
                        .bg(rgba((BG() << 8) | 0xd9))
                        .text_size(px(10.5))
                        .text_color(rgb(BLUE()))
                        .child(tr!("Vista previa en vivo", "Live preview")),
                )
            })
            .when(zoomed, |d| {
                d.child(
                    div()
                        .absolute()
                        .top(px(10.))
                        .left(px(10.))
                        .flex()
                        .items_center()
                        .gap(px(6.))
                        .child(
                            div()
                                .px(px(8.))
                                .py(px(3.))
                                .rounded(px(5.))
                                .bg(rgba((BG() << 8) | 0xd9))
                                .font_family(MONO)
                                .text_size(px(11.))
                                .text_color(rgb(TEXT()))
                                .child(format!("{:.0} %", self.zoom * 100.)),
                        )
                        .child(button("fit", tr!("Ajustar", "Fit")).h(px(24.)).on_click(cx.listener(|this, _, _, cx| {
                            this.zoom = 1.0;
                            this.pan = Point::default();
                            cx.notify();
                        }))),
                )
            })
            .children(thumbs.and_then(|(one, two, w, h)| {
                // real pixels: one image pixel per screen pixel (logical size = pixels / scale factor)
                let (w1, h1) = (w as f32 / scale, h as f32 / scale);
                // a corner card, never a cover: 2x only while both fit in ~30 % of the viewer, nothing past 20 % at 1x
                if w1.max(h1) > side * 0.2 {
                    return None;
                }
                let with_two = w1 * 3. + 24. <= side * 0.3 && h1 * 2. <= side * 0.3;
                Some(
                    div()
                        .absolute()
                        .right(px(10.))
                        .bottom(px(10.))
                        .flex()
                        .items_end()
                        .gap(px(8.))
                        .p(px(8.))
                        .rounded(px(8.))
                        .bg(rgba((BG() << 8) | 0xe6))
                        .border_1()
                        .border_color(rgb(BORDER()))
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .items_center()
                                .gap(px(4.))
                                .child(img(one).w(px(w1)).h(px(h1)))
                                .child(div().text_size(px(9.5)).text_color(rgb(MUTED_2())).child("1×")),
                        )
                        .when(with_two, |d| {
                            d.child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .items_center()
                                    .gap(px(4.))
                                    .child(img(two).w(px(w1 * 2.)).h(px(h1 * 2.)))
                                    .child(div().text_size(px(9.5)).text_color(rgb(MUTED_2())).child("2×")),
                            )
                        }),
                )
            }));
        let view = div()
            .relative()
            .flex_1()
            .min_h_0()
            .m(px(16.))
            .overflow_hidden()
            .flex()
            .items_center()
            .justify_center()
            .child(
                canvas(
                    move |b, window, _| {
                        if area.get() != Some(b) {
                            area.set(Some(b));
                            window.refresh();
                        }
                    },
                    |_, _, _, _| (),
                )
                .absolute()
                .size_full(),
            )
            .child(square);
        // results of the current / last job
        let strip = (self.job_results.len() > 1).then(|| {
            div().flex().gap(px(8.)).px(px(16.)).pb(px(4.)).children(self.job_results.iter().enumerate().filter_map(|(i, p)| {
                let it = self.history.iter().find(|x| &x.png == p)?;
                let sel = self.selected.as_ref() == Some(p);
                let p = p.clone();
                Some(
                    div()
                        .id(("job", i))
                        .w(px(64.))
                        .h(px(64.))
                        .rounded(px(6.))
                        .overflow_hidden()
                        .bg(rgb(CHECK_A()))
                        .cursor_pointer()
                        .border_2()
                        .border_color(rgb(if sel { ACCENT() } else { BORDER_SOFT() }))
                        .child(img(it.thumb.clone()).size_full().object_fit(ObjectFit::Contain))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.select(Some(p.clone()));
                            cx.notify();
                        })),
                )
            }))
        });
        div()
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .bg(rgb(BG()))
            .child(
                div()
                    .h(px(44.))
                    .flex_none()
                    .flex()
                    .items_center()
                    .px(px(18.))
                    .gap(px(10.))
                    .border_b_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .child(div().text_size(px(12.)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT())).child(if running {
                        tr!("Generando", "Generating")
                    } else if self.mode == Mode::Pose {
                        tr!("Vista previa", "Preview")
                    } else {
                        tr!("Resultado", "Result")
                    }))
                    .child(div().flex_1())
                    .child(div().font_family(MONO).text_size(px(11.)).text_color(rgb(MUTED())).child(header_right)),
            )
            .child(view)
            .children(strip)
            .when(!running && self.mode != Mode::Pose, |d| d.children(self.details(cx)))
    }

    /// Right-click menu of the selected history result: what the result card offers, plus the shortcuts.
    fn context_menu(&mut self, at: Point<Pixels>, cx: &mut Context<Self>) -> AnyElement {
        let (w, h) = (230., 6. * 32. + 22.);
        let (vw, vh) = self.viewport;
        let x = f32::from(at.x).min(vw - w - 8.);
        let y = f32::from(at.y).min(vh - h - 8.);
        let png = self.selected.clone();
        type Act = Box<dyn Fn(&mut Studio, &mut Context<Studio>)>;
        let items: Vec<(&'static str, &str, Option<&str>, Act)> = vec![
            ("image-plus", tr!("Usar como origen", "Use as source"), None, Box::new(|s, cx| s.use_as_source(cx))),
            ("rotate-ccw", tr!("Reutilizar ajustes", "Reuse settings"), None, Box::new(|s, cx| s.reuse(cx))),
            ("copy", tr!("Copiar imagen", "Copy image"), Some("Ctrl C"), Box::new(|s, _| s.copy_image(false))),
            ("maximize-2", tr!("Copiar ampliada", "Copy enlarged"), None, Box::new(|s, _| s.copy_image(true))),
            (
                "folder-open",
                tr!("Mostrar en la carpeta", "Show in folder"),
                None,
                Box::new(move |_, _| {
                    if let Some(p) = &png {
                        reveal_in_explorer(p)
                    }
                }),
            ),
            ("trash-2", tr!("Mover a la papelera", "Move to Recycle Bin"), Some(tr!("Supr", "Del")), Box::new(|s, cx| s.delete_selected(cx))),
        ];
        let rows: Vec<AnyElement> = items
            .into_iter()
            .enumerate()
            .map(|(i, (glyph, name, keys, act))| {
                let danger = glyph == "trash-2";
                div()
                    .kb(("ctx", i))
                    .h(px(32.))
                    .px(px(10.))
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .rounded(px(6.))
                    .cursor_pointer()
                    .hover(move |s| if danger { s.bg(rgba((DANGER() << 8) | 0x22)) } else { s.bg(rgb(HOVER())) })
                    .child(icon(glyph).size(px(14.)).text_color(rgb(if danger { DANGER() } else { MUTED() })))
                    .child(div().flex_1().text_size(px(12.5)).text_color(rgb(if danger { DANGER() } else { TEXT() })).child(name.to_string()))
                    .children(keys.map(kbd))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.ctx_menu = None;
                        act(this, cx);
                        cx.notify();
                    }))
                    .into_any_element()
            })
            .collect();
        div()
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .child(
                div()
                    .id("ctx-backdrop")
                    .absolute()
                    .top_0()
                    .left_0()
                    .size_full()
                    .on_mouse_down(
                        MouseButton::Right,
                        cx.listener(|this, _, _, cx| {
                            this.ctx_menu = None;
                            cx.notify();
                        }),
                    )
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.ctx_menu = None;
                        cx.notify();
                    })),
            )
            .child(
                div()
                    .id("ctx-menu")
                    .occlude()
                    .absolute()
                    .left(px(x))
                    .top(px(y))
                    .w(px(w))
                    .p(px(5.))
                    .flex()
                    .flex_col()
                    .gap(px(1.))
                    .rounded(px(9.))
                    .bg(rgb(PANEL()))
                    .border_1()
                    .border_color(rgb(BORDER()))
                    .shadow_lg()
                    .children(rows),
            )
            .into_any_element()
    }

    fn details(&mut self, cx: &mut Context<Self>) -> Option<AnyElement> {
        let it = self.selected_item()?;
        let j = &it.json;
        let (w, h) = it.sprite.dimensions();
        let mode = j["mode"].as_str().and_then(Mode::from_key);
        let mut facts: Vec<String> = vec![mode.map(|m| m.label().to_string()).unwrap_or_default(), format!("{w}×{h}")];
        facts.extend(j["seed"].as_i64().map(|v| format!("{} {v}", tr!("semilla", "seed"))));
        facts.extend(j["steps"].as_u64().map(|v| format!("{v} {}", tr!("pasos", "steps"))));
        facts.extend(
            j["lora_scale"].as_f64().filter(|v| (v - 1.).abs() > 0.001 && !j["lora"].as_str().unwrap_or("").is_empty()).map(|v| format!("LoRA {v:.2}")),
        );
        facts.extend(j["guidance"].as_f64().map(|v| format!("CFG {v:.1}")));
        facts.extend(j["strength"].as_f64().filter(|_| mode.is_some_and(|m| m.uses_source())).map(|v| format!("{} {v:.2}", tr!("intensidad", "strength"))));
        facts.extend(j["seconds"].as_f64().map(|v| format!("{v:.0} s")));
        let facts: Vec<String> = facts.into_iter().filter(|f| !f.is_empty()).collect();
        let base = j["base"].as_str().unwrap_or("").to_string();
        let turbo = base.contains("Turbo");
        let model = match j["version"].as_str() {
            Some(v) => format!("{base} · {v}"),
            None => base,
        };
        let prompt = j["prompt"].as_str().unwrap_or("").to_string();
        let png = it.png.clone();
        Some(
            div()
                .flex_none()
                .h(px(DETAILS_H))
                .overflow_hidden()
                .mx(px(16.))
                .mb(px(16.))
                .px(px(16.))
                .py(px(14.))
                .rounded(px(10.))
                .bg(rgb(PANEL()))
                .border_1()
                .border_color(rgb(BORDER_SOFT()))
                .flex()
                .flex_col()
                .gap(px(10.))
                .child(
                    div()
                        .flex()
                        .items_start()
                        .gap(px(10.))
                        .child(
                            div()
                                .id("detail-prompt")
                                .flex_1()
                                .max_h(px(19. * 2.))
                                .overflow_y_scroll()
                                .text_size(px(13.))
                                .line_height(px(19.))
                                .text_color(rgb(if prompt.is_empty() { MUTED_2() } else { TEXT() }))
                                .child(if prompt.is_empty() { tr!("Sin descripción", "No description").to_string() } else { prompt.clone() }),
                        )
                        .when(!prompt.is_empty(), |d| {
                            d.child(icon_btn("copy", "clipboard-copy").tooltip(tooltip(tr!("Copiar la descripción", "Copy the description"))).on_click(
                                cx.listener(move |this, _, _, cx| {
                                    cx.write_to_clipboard(gpui::ClipboardItem::new_string(prompt.clone()));
                                    this.say(tr!("Descripción copiada.", "Description copied."), Level::Info);
                                    cx.notify();
                                }),
                            ))
                        }),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(10.))
                        .min_w_0()
                        .when(!model.is_empty(), |d| {
                            d.child(
                                div()
                                    .flex_none()
                                    .flex()
                                    .items_center()
                                    .gap(px(5.))
                                    .h(px(22.))
                                    .px(px(8.))
                                    .rounded(px(6.))
                                    .bg(rgb(RAISED()))
                                    .when(turbo, |d| d.child(icon("zap").size(px(11.)).text_color(rgb(ACCENT()))))
                                    .child(div().text_size(px(11.)).text_color(rgb(TEXT())).child(model)),
                            )
                        })
                        .child(
                            div()
                                .id("detail-facts")
                                .flex_1()
                                .min_w_0()
                                .overflow_hidden()
                                .text_ellipsis()
                                .whitespace_nowrap()
                                .font_family(MONO)
                                .text_size(px(11.))
                                .text_color(rgb(MUTED()))
                                .tooltip(tooltip(&facts.join(" · ")))
                                .child(facts.join("  ·  ")),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(6.))
                        .child(
                            button_i("reuse", "rotate-ccw", tr!("Reutilizar ajustes", "Reuse settings")).on_click(cx.listener(|this, _, _, cx| this.reuse(cx))),
                        )
                        .child(
                            button_i("as-src", "image-plus", tr!("Usar como origen", "Use as source"))
                                .on_click(cx.listener(|this, _, _, cx| this.use_as_source(cx))),
                        )
                        .child(div().flex_1())
                        .child(
                            icon_btn("copy-img", "copy")
                                .w(px(30.))
                                .h(px(30.))
                                .tooltip(tooltip(tr!("Copiar imagen · Mayús: ampliada", "Copy image · Shift: enlarged")))
                                .on_click(cx.listener(|this, ev: &ClickEvent, _, cx| {
                                    this.copy_image(ev.modifiers().shift);
                                    cx.notify();
                                })),
                        )
                        .child(
                            icon_btn("reveal", "folder-open")
                                .w(px(30.))
                                .h(px(30.))
                                .tooltip(tooltip(tr!("Mostrar en la carpeta", "Show in folder")))
                                .on_click(cx.listener(move |_, _, _, _| reveal_in_explorer(&png))),
                        )
                        .child(
                            icon_btn("del", "trash-2")
                                .w(px(30.))
                                .h(px(30.))
                                .tooltip(tooltip(tr!("Mover a la papelera (Supr)", "Move to Recycle Bin (Delete)")))
                                .hover(|s| s.bg(rgba(0xe07a6a22)).border_color(rgb(DANGER())))
                                .on_click(cx.listener(|this, _, _, cx| this.delete_selected(cx))),
                        ),
                )
                .into_any_element(),
        )
    }

    fn history_panel(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let items: Vec<(usize, PathBuf, Arc<RenderImage>, &'static str, bool)> = self
            .visible_history()
            .into_iter()
            .enumerate()
            .map(|(i, it)| {
                (
                    i,
                    it.png.clone(),
                    it.thumb.clone(),
                    it.json["mode"].as_str().and_then(Mode::from_key).map(|m| m.glyph()).unwrap_or(""),
                    it.json["base"].as_str().is_some_and(|b| b.contains("Turbo")),
                )
            })
            .collect();
        let total = items.len();
        let has_any = !self.history.is_empty();
        div()
            .w(px(292.))
            .flex_none()
            .flex()
            .flex_col()
            .bg(rgb(PANEL()))
            .border_l_1()
            .border_color(rgb(BORDER_SOFT()))
            .child(
                div()
                    .h(px(44.))
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .px(px(14.))
                    .border_b_1()
                    .border_color(rgb(BORDER_SOFT()))
                    .child(div().text_size(px(12.)).font_weight(FontWeight::SEMIBOLD).child(tr!("Historial", "History")))
                    .child(div().text_size(px(11.)).text_color(rgb(MUTED_2())).child(format!("{total}")))
                    .child(div().flex_1())
                    .child(segmented([
                        seg("hist-all", tr!("Todo", "All"), !self.only_mode).h(px(22.)).text_size(px(11.)).on_click(cx.listener(|this, _, _, cx| {
                            this.only_mode = false;
                            cx.notify();
                        })),
                        seg("hist-mode", tr!("Este modo", "This mode"), self.only_mode).h(px(22.)).text_size(px(11.)).on_click(cx.listener(
                            |this, _, _, cx| {
                                this.only_mode = true;
                                cx.notify();
                            },
                        )),
                    ])),
            )
            .when(has_any, |d| d.child(div().px(px(12.)).pt(px(12.)).child(self.hist_query.clone())))
            .child(
                div().id("history").flex_1().min_h_0().overflow_y_scroll().p(px(12.)).child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap(px(8.))
                        .children(items.into_iter().map(|(i, png, thumb, glyph, turbo)| {
                            let sel = self.selected.as_ref() == Some(&png);
                            let thumb_drag = thumb.clone();
                            div()
                                .kb(("hist", i))
                                .relative()
                                .w(px(82.))
                                .h(px(82.))
                                .rounded(px(7.))
                                .overflow_hidden()
                                .bg(rgb(CHECK_A()))
                                .cursor_pointer()
                                .border_2()
                                .border_color(rgb(if sel { ACCENT() } else { BORDER_SOFT() }))
                                .hover(|s| s.border_color(rgb(ACCENT_DIM())))
                                .child(img(thumb).size_full().p(px(4.)).object_fit(ObjectFit::Contain))
                                .when(!glyph.is_empty(), |d| d.child(icon(glyph).absolute().top(px(4.)).right(px(4.)).size(px(11.)).text_color(rgb(MUTED()))))
                                // generated with the Turbo base: a flash in the bottom corner
                                .when(turbo, |d| {
                                    d.child(
                                        div()
                                            .absolute()
                                            .bottom(px(3.))
                                            .right(px(3.))
                                            .p(px(2.))
                                            .rounded(px(4.))
                                            .bg(rgba((BG() << 8) | 0xd9))
                                            .child(icon("zap").size(px(11.)).text_color(rgb(ACCENT()))),
                                    )
                                })
                                .on_drag(DraggedSprite { png: png.clone(), thumb: thumb_drag.clone() }, |d, _, _, cx| {
                                    cx.new(|_| DragPreview { thumb: d.thumb.clone() })
                                })
                                .on_mouse_down(
                                    MouseButton::Right,
                                    cx.listener({
                                        let png = png.clone();
                                        move |this, ev: &MouseDownEvent, _, cx| {
                                            this.select(Some(png.clone()));
                                            this.ctx_menu = Some(ev.position);
                                            cx.stop_propagation();
                                            cx.notify();
                                        }
                                    }),
                                )
                                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                    this.select(Some(png.clone()));
                                    cx.notify();
                                }))
                        }))
                        .when(total == 0 && has_any, |d| {
                            d.child(
                                div()
                                    .w(px(266.))
                                    .pt(px(30.))
                                    .flex()
                                    .justify_center()
                                    .text_size(px(12.))
                                    .text_color(rgb(MUTED_2()))
                                    .child(tr!("Nada coincide", "Nothing matches")),
                            )
                        })
                        .when(total == 0 && !has_any, |d| {
                            d.child(
                                div()
                                    .w(px(266.))
                                    .pt(px(40.))
                                    .px(px(12.))
                                    .flex()
                                    .flex_col()
                                    .items_center()
                                    .gap(px(8.))
                                    .child(icon("history").size(px(22.)).text_color(rgb(MUTED_2())))
                                    .child(div().text_size(px(12.5)).text_color(rgb(MUTED())).child(tr!("Aún no hay resultados", "No results yet")))
                                    .child(div().w_full().text_size(px(11.)).line_height(px(16.)).text_color(rgb(MUTED_2())).text_center().child(tr!(
                                        "Lo que generes aparece aquí. Arrastra uno al panel izquierdo para usarlo como origen.",
                                        "What you make shows up here. Drag one onto the left panel to use it as a source."
                                    ))),
                            )
                        }),
                ),
            )
    }

    fn statusbar(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        if let Some((_, lvl, t)) = &self.status
            && *lvl != Level::Error
            && t.elapsed() > Duration::from_secs(12)
            && !self.running
            && !self.pending
        {
            self.status = None;
        }
        let (msg, col) = match &self.status {
            Some((m, Level::Error, _)) => (m.clone(), DANGER()),
            Some((m, Level::Ok, _)) => (m.clone(), ACCENT()),
            Some((m, Level::Info, _)) => (m.clone(), MUTED()),
            None => (String::new(), MUTED_2()),
        };
        let first = msg.lines().next().unwrap_or("").to_string();
        let out = crate::settings::display_path(&self.model.out_dir);
        let out_tip = tr!(format!("Abrir {out}"), format!("Open {out}"));
        let out_name = self.model.out_dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let (dot, gpu) = self.gpu_status();
        let out_dir = self.model.out_dir.clone();
        let sys = self.sys.unwrap_or_default();
        let ram_tip = tr!(
            format!("Memoria RAM del sistema en uso · Dotsmith: {}", crate::download::gb(sys.app_ram)),
            format!("System RAM in use · Dotsmith: {}", crate::download::gb(sys.app_ram))
        );
        let vram_tip = tr!(
            "VRAM de la GPU en uso, contando todas las aplicaciones. Si se llena, la generación se detiene con un aviso.",
            "GPU VRAM in use, counting every app. If it fills up, generation stops with a warning."
        );
        div()
            .h(px(28.))
            .flex_none()
            .flex()
            .items_center()
            .gap(px(2.))
            .pl(px(14.))
            .pr(px(6.))
            .bg(rgb(BG()))
            .border_t_1()
            .border_color(rgb(BORDER_SOFT()))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .child(div().min_w_0().overflow_hidden().text_ellipsis().whitespace_nowrap().text_size(px(11.5)).text_color(rgb(col)).child(first))
                    .when(self.deleted.is_some(), |d| {
                        d.child(
                            status_item("undo")
                                .cursor_pointer()
                                .border_1()
                                .border_color(rgb(BORDER()))
                                .child(icon("rotate-ccw").size(px(12.)).text_color(rgb(TEXT())))
                                .child(div().text_size(px(11.)).text_color(rgb(TEXT())).child(tr!("Deshacer", "Undo")))
                                .child(div().font_family(MONO).text_size(px(10.)).text_color(rgb(MUTED_2())).child("Ctrl Z"))
                                .on_click(cx.listener(|this, _, _, cx| this.undo_delete(cx))),
                        )
                    }),
            )
            .when(self.sys.is_some(), |d| d.child(mem_meter("mon-ram", "RAM", sys.ram_used, sys.ram_total, &ram_tip)).child(status_sep()))
            .when_some(sys.vram, |d, (used, total)| d.child(mem_meter("mon-vram", "VRAM", used, total, vram_tip)).child(status_sep()))
            .child(
                status_item("gpu-state")
                    .child(div().w(px(7.)).h(px(7.)).rounded_full().bg(rgb(dot)))
                    .child(div().text_size(px(11.)).text_color(rgb(MUTED())).child(gpu)),
            )
            .child(status_sep())
            .child(
                status_item("out-dir")
                    .cursor_pointer()
                    .tooltip(tooltip(&out_tip))
                    .child(icon("folder-open").size(px(12.)).text_color(rgb(MUTED_2())))
                    .child(div().text_size(px(11.)).text_color(rgb(MUTED())).child(out_name))
                    .on_click(cx.listener(move |_, _, _, cx| {
                        let _ = std::fs::create_dir_all(&out_dir);
                        cx.open_with_system(&out_dir);
                    })),
            )
    }

    fn error_detail(&self) -> Option<AnyElement> {
        let (msg, lvl, _) = self.status.as_ref()?;
        if *lvl != Level::Error || msg.lines().count() < 2 {
            return None;
        }
        Some(
            div()
                .absolute()
                .bottom(px(36.))
                .right(px(308.))
                .w(px(560.))
                .max_h(px(260.))
                .p(px(12.))
                .rounded(px(8.))
                .bg(rgb(0x2a1d19))
                .border_1()
                .border_color(rgb(DANGER()))
                .font_family(MONO)
                .text_size(px(10.5))
                .text_color(rgb(0xf3c6bd))
                .overflow_hidden()
                .child(msg.lines().rev().take(14).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n"))
                .into_any_element(),
        )
    }
}

impl Render for Studio {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        TAB_OFF.store(false, std::sync::atomic::Ordering::Relaxed);
        kb_refill(cx);
        self.scale_factor = window.scale_factor();
        self.hist_query_text = self.hist_query.read(cx).text();
        self.maximized = window.is_maximized();
        let vp = window.viewport_size();
        self.viewport = (f32::from(vp.width), f32::from(vp.height));
        if !self.maximized {
            self.restore_size = self.viewport;
        }
        for im in self.trash.drain(..) {
            let _ = window.drop_image(im);
        }
        let popover = self.model_open.then(|| self.model_picker(cx));
        let menu = self.menu_open.then(|| self.menu_dropdown(cx));
        let prompt_modal = self.prompt_modal.then(|| self.prompt_modal_view(cx));
        let settings = self.settings_open.then(|| self.settings_view(cx));
        let palette = self.palette_open.then(|| self.palette_view(cx));
        let ctx_menu = self.ctx_menu.map(|p| self.context_menu(p, cx));
        let hub = self.hub_open.then(|| self.hub_view(cx));
        let downloads = self.downloads_open.then(|| self.downloads_view(cx));
        let err = self.error_detail();
        let welcome = self.welcome_visible();
        // the modals above are built; the background below loses its tab stops while one is open
        let modal = self.model_open || self.prompt_modal || self.settings_open || self.palette_open || self.hub_open || self.ctx_menu.is_some();
        TAB_OFF.store(modal, std::sync::atomic::Ordering::Relaxed);
        // the background's text fields leave the tab order too
        for input in [self.form().prompt.clone(), self.form().seed.clone(), self.hist_query.clone()] {
            let _ = input.read(cx).focus_handle.clone().tab_stop(!modal);
        }
        div()
            .id("root")
            .key_context("Studio")
            .track_focus(&self.focus)
            .on_action(cx.listener(Self::generate))
            .on_action(cx.listener(Self::stop))
            .on_action(cx.listener(Self::paste_image))
            .on_action(cx.listener(Self::copy_image_action))
            .on_action(cx.listener(|this, _: &DeleteSelected, _, cx| this.delete_selected(cx)))
            .on_action(cx.listener(|this, _: &UndoDelete, _, cx| this.undo_delete(cx)))
            .on_action(cx.listener(|_, _: &FocusNext, w, cx| {
                w.focus_next();
                cx.notify();
            }))
            .on_action(cx.listener(|_, _: &FocusPrev, w, cx| {
                w.focus_prev();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &PrevResult, _, cx| this.step_selection(-1, cx)))
            .on_action(cx.listener(|this, _: &NextResult, _, cx| this.step_selection(1, cx)))
            .on_action(cx.listener(|this, _: &OpenSettings, _, cx| {
                this.settings_open = true;
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &OpenPalette, w, cx| if this.palette_open { this.close_palette(w, cx) } else { this.open_palette(w, cx) }))
            .on_action(cx.listener(|this, _: &OpenModelPicker, _, cx| {
                if this.model_open {
                    this.model_open = false
                } else {
                    this.open_model_picker()
                }
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &crate::input::Paste, w, cx| this.paste_image(&PasteImage, w, cx)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, window, _| {
                    // text fields stop the event; any other click gives the keys back to the studio (Ctrl+V pastes an image)
                    window.focus(&this.focus);
                }),
            )
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                if let Some(p) = paths.paths().first() {
                    if !this.mode.uses_source() {
                        this.mode = Mode::Variants;
                    }
                    this.set_source(p.clone(), cx);
                }
            }))
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(BG()))
            .text_color(rgb(TEXT()))
            .font_family(FONT)
            .child(self.titlebar(cx))
            .when(welcome, |d| d.child(self.welcome_view(cx)))
            .when(!welcome, |d| {
                d.child(div().flex_1().min_h_0().flex().child(self.rail(cx)).child(self.controls(cx)).child(self.stage(cx)).child(self.history_panel(cx)))
                    .child(self.statusbar(cx))
            })
            .children(err)
            .children(popover)
            .children(prompt_modal)
            .children(menu)
            .children(hub)
            .children(downloads)
            .children(settings)
            .children(ctx_menu)
            .children(palette)
    }
}

/// Opens Explorer with the file selected. gpui's reveal_path did nothing on Windows; explorer needs a backslash,
/// absolute path passed raw after `/select,` (std's argument quoting would break it on paths with spaces).
fn reveal_in_explorer(path: &std::path::Path) {
    let full = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let text = full.to_string_lossy().trim_start_matches(r"\\?\").replace('/', "\\");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = std::process::Command::new("explorer").raw_arg(format!("/select,\"{text}\"")).spawn();
    }
    #[cfg(not(windows))]
    let _ = std::process::Command::new("xdg-open").arg(full.parent().unwrap_or(&full)).spawn();
}

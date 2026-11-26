//! Lucide icons (lucide-static, ISC, assets/icons/LICENSE), embedded in the exe and served to gpui's `svg()`.
//! gpui draws an SVG as a mask tinted with the element's own text colour, so set `.text_color` on the icon itself.
use gpui::{AssetSource, Result, SharedString, Svg, prelude::*, px, svg};
use std::borrow::Cow;

macro_rules! icons {
    ($($name:literal),* $(,)?) => {
        const ICONS: &[(&str, &[u8])] = &[$(($name, include_bytes!(concat!("../assets/icons/", $name, ".svg")))),*];
    };
}

icons!(
    "box",
    "brush",
    "chevron-down",
    "clipboard-copy",
    "copy",
    "folder-open",
    "grid-3x3",
    "history",
    "image",
    "image-plus",
    "layers",
    "loader-circle",
    "maximize-2",
    "minus",
    "plus",
    "rotate-3d",
    "rotate-ccw",
    "square",
    "trash-2",
    "x",
    "zap",
    "person-standing",
    "play",
    "pause",
    "footprints",
    "brick-wall",
    "scissors",
    "eraser",
    "settings",
    "check",
    "circle-alert",
    "triangle-alert",
    "cpu",
    "info",
    "file-text",
    "download",
    "star",
    "circle-check-big",
    "external-link",
    "key-round",
    "memory-stick",
    "gpu",
    "chevron-right",
    "keyboard",
    "wrench",
    "type",
    "search",
);

pub struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        let name = path.strip_prefix("icons/").and_then(|p| p.strip_suffix(".svg"));
        Ok(name.and_then(|n| ICONS.iter().find(|i| i.0 == n)).map(|i| Cow::Borrowed(i.1)))
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        Ok(ICONS.iter().map(|i| format!("icons/{}.svg", i.0)).filter(|p| p.starts_with(path)).map(SharedString::from).collect())
    }
}

/// The loader icon turning (one turn per second): something is in progress and the app is alive.
pub fn spinner(id: impl Into<gpui::ElementId>, size: f32, color: u32) -> gpui::AnimationElement<Svg> {
    use gpui::{Animation, AnimationExt, Transformation, percentage};
    icon("loader-circle")
        .size(px(size))
        .text_color(gpui::rgb(color))
        .with_animation(id, Animation::new(std::time::Duration::from_secs(1)).repeat(), |s, d| s.with_transformation(Transformation::rotate(percentage(d))))
}

/// A 16 px Lucide icon; resize with `.size(..)` and colour with `.text_color(..)`.
pub fn icon(name: &'static str) -> Svg {
    svg().path(SharedString::from(format!("icons/{name}.svg"))).flex_none().size(px(16.))
}

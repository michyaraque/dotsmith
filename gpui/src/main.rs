#![windows_subsystem = "windows"]
mod anim;
mod aurora;
mod backend;
mod clipboard;
mod download;
mod hf;
mod i18n;
mod icons;
mod input;
mod instance;
mod jobs;
mod log;
mod pack_split;
mod pixels;
mod rig;
mod settings;
mod studio;
mod sysmon;
mod theme;

use gpui::{App, AppContext, Application, Bounds, TitlebarOptions, WindowBounds, WindowOptions, px, size};

fn main() {
    if !instance::acquire() {
        return;
    }
    log::init();
    let settings = settings::Settings::load();
    i18n::set(settings.lang());
    theme::set_light(settings.light());
    settings.apply_to_engine();
    Application::new().with_assets(icons::Assets).run(move |cx: &mut App| {
        input::bind_keys(cx);
        studio::bind_keys(cx);
        // the default, the saved and the minimum size all fit the screen (a 1080p laptop at 150 % is 1280 x 720)
        let screen = cx.primary_display().map(|d| d.bounds().size).map(|s| (f32::from(s.width), f32::from(s.height)));
        let fit = |w: f32, h: f32| screen.map_or((w, h), |(sw, sh)| (w.min(sw * 0.94), h.min(sh * 0.88)));
        let (min_w, min_h) = fit(1180., 600.);
        let saved = settings.window.clone().filter(|w| w.width >= min_w && w.height >= min_h);
        let (w, h) = saved.as_ref().map_or((1500., 940.), |w| (w.width, w.height));
        let (w, h) = fit(w, h);
        let bounds = Bounds::centered(None, size(px(w), px(h)), cx);
        let window_bounds = if saved.is_some_and(|w| w.maximized) { WindowBounds::Maximized(bounds) } else { WindowBounds::Windowed(bounds) };
        cx.open_window(
            WindowOptions {
                window_bounds: Some(window_bounds),
                titlebar: Some(TitlebarOptions { title: Some("Dotsmith".into()), appears_transparent: true, ..Default::default() }),
                window_min_size: Some(size(px(min_w), px(min_h))),
                ..Default::default()
            },
            |window, cx| cx.new(|cx| studio::Studio::new(settings, window, cx)),
        )
        .unwrap();
        cx.on_window_closed(|cx| cx.quit()).detach();
        cx.activate(true);
    });
}

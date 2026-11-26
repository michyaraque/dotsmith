//! Palette, dark or light. Dark: cozy warm charcoal with a cream accent. Light: warm paper with charcoal primary
//! buttons. Every colour is a function so the whole interface follows a theme switch on the next frame.
use std::sync::atomic::{AtomicBool, Ordering};

static LIGHT: AtomicBool = AtomicBool::new(false);

pub fn set_light(on: bool) {
    LIGHT.store(on, Ordering::Relaxed);
}

macro_rules! palette {
    ($($(#[$doc:meta])* $name:ident = $dark:expr, $light:expr;)*) => {
        $(
            $(#[$doc])*
            #[allow(non_snake_case)]
            #[inline]
            pub fn $name() -> u32 {
                if LIGHT.load(Ordering::Relaxed) { $light } else { $dark }
            }
        )*
    };
}

palette! {
    BG = 0x1c1a17, 0xf3f0ea;
    PANEL = 0x23211d, 0xfaf8f4;
    RAISED = 0x2b2824, 0xebe6de;
    HOVER = 0x34302b, 0xe2dcd2;
    INPUT_BG = 0x1f1d1a, 0xffffff;
    BORDER = 0x3d3832, 0xd5cec2;
    BORDER_SOFT = 0x2e2b26, 0xe4ded4;
    TEXT = 0xede6da, 0x2a2621;
    MUTED = 0xaaa194, 0x635c52;
    MUTED_2 = 0x958c7f, 0x857e73;
    /// primary buttons, active chips, progress
    ACCENT = 0xeee2cb, 0x2e2a25;
    ACCENT_HOVER = 0xf7eedc, 0x45403a;
    /// focus ring
    ACCENT_DIM = 0x9c8f78, 0xa39680;
    ACCENT_INK = 0x1f1b15, 0xfaf8f4;
    /// live / ready indicators: soft amber (name kept for the call sites)
    BLUE = 0xe5a95f, 0xb8741a;
    DANGER = 0xe07a6a, 0xc0503d;
    WARN = 0xe8c06a, 0xa8780c;
    GREEN = 0x9ccf8f, 0x4a8a3a;
    CYAN = 0x86c8d8, 0x2c8299;
    CHECK_A = 0x211f1b, 0xefebe4;
    CHECK_B = 0x282521, 0xe5e0d7;
    /// modal backdrop (rgba)
    VEIL = 0x0b0a08b8, 0x2a262159;
}

pub const FONT: &str = "Segoe UI";
pub const MONO: &str = "Cascadia Mono";

/// Whether Windows apps are set to light mode (Settings > Personalisation > Colours).
pub fn system_prefers_light() -> bool {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn RegGetValueW(hkey: isize, sub: *const u16, value: *const u16, flags: u32, ty: *mut u32, data: *mut u8, len: *mut u32) -> i32;
        }
        const HKEY_CURRENT_USER: isize = 0x8000_0001u32 as i32 as isize;
        const RRF_RT_REG_DWORD: u32 = 0x10;
        let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
        let (sub, value) = (wide(r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"), wide("AppsUseLightTheme"));
        let mut data: u32 = 0;
        let mut len: u32 = 4;
        let ok = unsafe {
            RegGetValueW(HKEY_CURRENT_USER, sub.as_ptr(), value.as_ptr(), RRF_RT_REG_DWORD, std::ptr::null_mut(), (&mut data as *mut u32).cast(), &mut len)
        } == 0;
        ok && data == 1
    }
    #[cfg(not(windows))]
    {
        false
    }
}

//! Interface language. Strings are written in place with both languages: `tr!("Generar", "Generate")`.
use std::sync::atomic::{AtomicBool, Ordering};

static EN: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lang {
    Es,
    En,
}

impl Lang {
    pub const ALL: [Lang; 2] = [Lang::En, Lang::Es];

    pub fn code(self) -> &'static str {
        match self {
            Lang::Es => "es",
            Lang::En => "en",
        }
    }

    pub fn from_code(c: &str) -> Option<Lang> {
        Lang::ALL.into_iter().find(|l| l.code() == c)
    }

    pub fn native(self) -> &'static str {
        match self {
            Lang::Es => "Español",
            Lang::En => "English",
        }
    }
}

pub fn set(l: Lang) {
    EN.store(l == Lang::En, Ordering::Relaxed);
}

#[inline]
pub fn en() -> bool {
    EN.load(Ordering::Relaxed)
}

/// Spanish for any Spanish Windows display language, English otherwise.
pub fn system() -> Lang {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetUserDefaultUILanguage() -> u16;
        }
        const LANG_SPANISH: u16 = 0x0a;
        if unsafe { GetUserDefaultUILanguage() } & 0x3ff == LANG_SPANISH { Lang::Es } else { Lang::En }
    }
    #[cfg(not(windows))]
    {
        if std::env::var("LANG").unwrap_or_default().starts_with("es") { Lang::Es } else { Lang::En }
    }
}

#[macro_export]
macro_rules! tr {
    ($es:expr, $en:expr $(,)?) => {
        if $crate::i18n::en() { $en } else { $es }
    };
}

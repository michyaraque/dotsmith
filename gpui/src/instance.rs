//! One Dotsmith per user session: two would each put a model on the GPU and run out of memory. A second launch brings
//! the running window forward and exits. DOTSMITH_MULTI_INSTANCE=1 lifts the limit (isolated screenshots, development).

/// True when this process may run: it is the first instance, or several are allowed.
pub fn acquire() -> bool {
    std::env::var("DOTSMITH_MULTI_INSTANCE").is_ok_and(|v| v == "1") || acquire_named("Local\\Dotsmith.SingleInstance", "Dotsmith")
}

/// Holds the named mutex for the life of the process, or brings the window titled `title` forward when another
/// process holds it.
fn acquire_named(mutex: &str, title: &str) -> bool {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn CreateMutexW(attrs: *const std::ffi::c_void, owner: i32, name: *const u16) -> isize;
            fn GetLastError() -> u32;
            fn FindWindowW(class: *const u16, title: *const u16) -> isize;
            fn ShowWindow(hwnd: isize, cmd: i32) -> i32;
            fn SetForegroundWindow(hwnd: isize) -> i32;
        }
        const ERROR_ALREADY_EXISTS: u32 = 183;
        const SW_RESTORE: i32 = 9;
        let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
        let name = wide(mutex);
        let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
        if handle != 0 && unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
            let title = wide(title);
            let hwnd = unsafe { FindWindowW(std::ptr::null(), title.as_ptr()) };
            if hwnd != 0 {
                unsafe {
                    ShowWindow(hwnd, SW_RESTORE);
                    SetForegroundWindow(hwnd);
                }
            }
            return false;
        }
    }
    let _ = (mutex, title);
    true
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_second_holder_is_refused() {
        let id = std::process::id();
        let (mutex, title) = (format!("Local\\Dotsmith.Test.{id}"), format!("dotsmith-test-no-window-{id}"));
        assert!(super::acquire_named(&mutex, &title));
        assert!(!super::acquire_named(&mutex, &title));
    }
}

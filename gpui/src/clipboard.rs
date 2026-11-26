//! Copy an image to the Windows clipboard in the formats other apps read: "PNG" (keeps alpha: browsers, Discord,
//! Photoshop, Aseprite) and CF_DIBV5 (32-bit with alpha: Paint, Office, most editors; Windows synthesizes CF_DIB and
//! CF_BITMAP from it). gpui's own clipboard writes only "PNG", which Paint and many others ignore.
use image::RgbaImage;

/// Encode `im` and put it on the clipboard. Err carries a short reason for the status line.
pub fn copy_image(im: &RgbaImage) -> Result<(), String> {
    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(im.clone()).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).map_err(|e| e.to_string())?;
    platform::set(&png, &dibv5(im))
}

/// BITMAPV5HEADER (124 bytes, BI_BITFIELDS with an alpha mask, sRGB) + bottom-up BGRA rows.
fn dibv5(im: &RgbaImage) -> Vec<u8> {
    let (w, h) = im.dimensions();
    let mut v = Vec::with_capacity(124 + (w * h * 4) as usize);
    let u32le = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
    u32le(&mut v, 124); // bV5Size
    v.extend_from_slice(&(w as i32).to_le_bytes());
    v.extend_from_slice(&(h as i32).to_le_bytes()); // positive = bottom-up
    v.extend_from_slice(&1u16.to_le_bytes()); // planes
    v.extend_from_slice(&32u16.to_le_bytes()); // bit count
    u32le(&mut v, 3); // BI_BITFIELDS
    u32le(&mut v, w * h * 4); // image size
    u32le(&mut v, 2835); // 72 dpi
    u32le(&mut v, 2835);
    u32le(&mut v, 0); // colours used
    u32le(&mut v, 0); // important
    u32le(&mut v, 0x00ff_0000); // red mask
    u32le(&mut v, 0x0000_ff00); // green
    u32le(&mut v, 0x0000_00ff); // blue
    u32le(&mut v, 0xff00_0000); // alpha
    u32le(&mut v, 0x7352_4742); // LCS_sRGB
    v.extend_from_slice(&[0u8; 36]); // endpoints
    u32le(&mut v, 0); // gamma r, g, b
    u32le(&mut v, 0);
    u32le(&mut v, 0);
    u32le(&mut v, 4); // LCS_GM_IMAGES
    u32le(&mut v, 0); // profile data, size, reserved
    u32le(&mut v, 0);
    u32le(&mut v, 0);
    debug_assert_eq!(v.len(), 124);
    for y in (0..h).rev() {
        for x in 0..w {
            let [r, g, b, a] = im.get_pixel(x, y).0;
            // fully transparent pixels carry no colour: apps that ignore alpha then show black, not stray RGB
            if a == 0 { v.extend_from_slice(&[0, 0, 0, 0]) } else { v.extend_from_slice(&[b, g, r, a]) }
        }
    }
    v
}

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;

    const CF_DIBV5: u32 = 17;
    const GMEM_MOVEABLE: u32 = 0x0002;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn OpenClipboard(owner: *mut c_void) -> i32;
        fn EmptyClipboard() -> i32;
        fn CloseClipboard() -> i32;
        fn SetClipboardData(format: u32, mem: *mut c_void) -> *mut c_void;
        fn RegisterClipboardFormatW(name: *const u16) -> u32;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalAlloc(flags: u32, bytes: usize) -> *mut c_void;
        fn GlobalLock(mem: *mut c_void) -> *mut c_void;
        fn GlobalUnlock(mem: *mut c_void) -> i32;
        fn GlobalFree(mem: *mut c_void) -> *mut c_void;
    }

    /// One clipboard format from a byte buffer; on success the clipboard owns the memory.
    unsafe fn put(format: u32, data: &[u8]) -> bool {
        unsafe {
            let mem = GlobalAlloc(GMEM_MOVEABLE, data.len());
            if mem.is_null() {
                return false;
            }
            let p = GlobalLock(mem) as *mut u8;
            if p.is_null() {
                GlobalFree(mem);
                return false;
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
            GlobalUnlock(mem);
            if SetClipboardData(format, mem).is_null() {
                GlobalFree(mem);
                return false;
            }
            true
        }
    }

    pub fn set(png: &[u8], dib: &[u8]) -> Result<(), String> {
        unsafe {
            // another app may hold the clipboard for a moment: retry briefly
            let mut opened = false;
            for _ in 0..10 {
                if OpenClipboard(std::ptr::null_mut()) != 0 {
                    opened = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(15));
            }
            if !opened {
                return Err(crate::tr!("el portapapeles está ocupado por otra aplicación", "another application is using the clipboard").into());
            }
            EmptyClipboard();
            let name: Vec<u16> = "PNG".encode_utf16().chain(Some(0)).collect();
            let png_ok = put(RegisterClipboardFormatW(name.as_ptr()), png);
            let dib_ok = put(CF_DIBV5, dib);
            CloseClipboard();
            if png_ok || dib_ok { Ok(()) } else { Err(crate::tr!("Windows rechazó los datos", "Windows rejected the data").into()) }
        }
    }
}

#[cfg(not(windows))]
mod platform {
    pub fn set(_: &[u8], _: &[u8]) -> Result<(), String> {
        Err(crate::tr!("solo disponible en Windows", "only available on Windows").into())
    }
}

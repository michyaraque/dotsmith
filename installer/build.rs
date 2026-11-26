//! The setup program carries the app's version (gpui/Cargo.toml, the one place it lives) in its code and in its file
//! properties. Its manifest is gpui's (DPI awareness); being 64-bit, Windows never elevates it for its name.
fn main() {
    println!("cargo:rerun-if-changed=../gpui/Cargo.toml");
    println!("cargo:rerun-if-changed=setup.rc");
    let manifest = std::fs::read_to_string("../gpui/Cargo.toml").unwrap();
    let version = manifest.lines().find_map(|l| l.strip_prefix("version = \"")?.strip_suffix('"')).expect("version in gpui/Cargo.toml").to_string();
    println!("cargo:rustc-env=DOTSMITH_VERSION={version}");
    let part = |i: usize| version.split('.').nth(i).unwrap_or("0").to_string();
    let macros = [
        format!("DOTSMITH_MAJOR={}", part(0)),
        format!("DOTSMITH_MINOR={}", part(1)),
        format!("DOTSMITH_PATCH={}", part(2)),
        format!("DOTSMITH_VERSION=\"{version}\""),
    ];
    embed_resource::compile("setup.rc", &macros).manifest_optional().unwrap();
}

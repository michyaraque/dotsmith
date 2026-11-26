fn main() {
    println!("cargo:rerun-if-changed=assets/dotsmith.rc");
    println!("cargo:rerun-if-changed=assets/dotsmith.ico");
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    let part = |k: &str| std::env::var(k).unwrap();
    let macros = [
        format!("DOTSMITH_MAJOR={}", part("CARGO_PKG_VERSION_MAJOR")),
        format!("DOTSMITH_MINOR={}", part("CARGO_PKG_VERSION_MINOR")),
        format!("DOTSMITH_PATCH={}", part("CARGO_PKG_VERSION_PATCH")),
        format!("DOTSMITH_VERSION=\"{version}\""),
    ];
    embed_resource::compile("assets/dotsmith.rc", &macros).manifest_optional().unwrap();

    // The window must open on PCs without an NVIDIA driver or CUDA DLLs: they load on first use, after
    // dotsmith_engine::gpu::probe has checked that they are there.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        for dll in ["nvcuda.dll", "cublas64_13.dll", "cublasLt64_13.dll", "curand64_10.dll", "nvrtc64_130_0.dll"] {
            println!("cargo:rustc-link-arg-bins=/DELAYLOAD:{dll}");
        }
        println!("cargo:rustc-link-arg-bins=delayimp.lib");
    }
}

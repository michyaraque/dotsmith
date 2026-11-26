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
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let runtime = if cuda_major() >= 13 {
            ["cublas64_13.dll", "cublasLt64_13.dll", "curand64_10.dll", "nvrtc64_130_0.dll"]
        } else {
            ["cublas64_12.dll", "cublasLt64_12.dll", "curand64_10.dll", "nvrtc64_120_0.dll"]
        };
        for dll in std::iter::once("nvcuda.dll").chain(runtime) {
            println!("cargo:rustc-link-arg-bins=/DELAYLOAD:{dll}");
        }
        println!("cargo:rustc-link-arg-bins=delayimp.lib");
    }
}

/// Major version of the CUDA toolkit on the PATH (the one the engine builds against).
fn cuda_major() -> u32 {
    let out = std::process::Command::new("nvcc").arg("--version").output().map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    out.split("release ").nth(1).and_then(|r| r.split('.').next()).and_then(|m| m.trim().parse().ok()).unwrap_or(13)
}

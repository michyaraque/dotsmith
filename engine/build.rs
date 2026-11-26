fn main() {
    // a C dependency asks for the static CRT (LNK4098); the binaries link the DLL one
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/NODEFAULTLIB:LIBCMT");
    }
    // the compute capability candle's kernels are built for is the oldest card this build runs on
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let cap = std::env::var("CUDA_COMPUTE_CAP").ok().and_then(|c| c.trim().parse::<u32>().ok()).unwrap_or(80);
    println!("cargo:rustc-env=DOTSMITH_MIN_COMPUTE={cap}");
    println!("cargo:rustc-env=DOTSMITH_CUDA_MAJOR={}", cuda_major());
}

/// Major version of the CUDA toolkit on the PATH (the one cudarc and candle build against).
fn cuda_major() -> u32 {
    let out = std::process::Command::new("nvcc").arg("--version").output().map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    out.split("release ").nth(1).and_then(|r| r.split('.').next()).and_then(|m| m.trim().parse().ok()).unwrap_or(13)
}

fn main() {
    // a C dependency asks for the static CRT (LNK4098); the binaries link the DLL one
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/NODEFAULTLIB:LIBCMT");
    }
}

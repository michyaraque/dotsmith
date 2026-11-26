# Releasing

Versions follow [SemVer](https://semver.org). The version lives in `gpui/Cargo.toml`. The exe's file properties, the installer and the archive names are all built from it.

1. Update `version` in `gpui/Cargo.toml` and add the section to `CHANGELOG.md`.
2. Run the checks:
   ```powershell
   cargo clippy --workspace --all-targets --release -- -D warnings
   cargo test --workspace --release
   cargo test --workspace --release -- --ignored
   ```
3. Build the release:
   ```powershell
   .\gpui\scripts\package.ps1
   ```
4. Smoke-test `dist\Dotsmith\dotsmith.exe` with the CUDA toolkit off the `PATH`. **Settings > GPU & memory** must show the card, which proves the bundled DLLs load.
5. Commit it as `chore: release 1.2.3` and push to `master`. The `Release` workflow sees a version with no release yet, builds both packages (standard, and legacy for cards before RTX 30), tags `v1.2.3` and publishes **Dotsmith v1.2.3** with that version's `CHANGELOG.md` section, what to download, `SHA256SUMS.txt` and GitHub's generated changelog. Pushes that keep the version publish nothing.

## What a release contains

| File | Purpose |
| --- | --- |
| `dotsmith.exe` | The app (icon, version info, CUDA DLLs delay-loaded) |
| `cublas64_13.dll`, `cublasLt64_13.dll`, `curand64_10.dll`, `nvrtc64_130_0.dll`, `nvrtc-builtins64_131.dll` | CUDA 13 runtime |
| `vcruntime140.dll` | MSVC runtime |
| `README.md`, `licenses/` | Docs and third-party licenses |

Model weights are never bundled.

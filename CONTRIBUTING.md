# Contributing

Thanks for helping with Dotsmith. This page covers how to build, what a good change looks like, and how to send it.

## Setup

Follow [Building from source](README.md#building-from-source), including the two tool paths in `%USERPROFILE%\.cargo\config.toml`. You need a CUDA-capable RTX GPU only for the end-to-end tests; everything else builds and tests without model files.

## Before you open a pull request

```powershell
cargo fmt --all
cargo clippy --workspace --all-targets --release -- -D warnings
cargo test --workspace --release
```

CI runs the same three checks; the build must produce no warnings.

- If you touched generation, also run `cargo test --workspace --release -- --ignored` (it generates one sprite on your GPU).
- For UI changes, attach before / after screenshots. `dotsmith.exe` opens a specific view with `--open-settings <general|models|gpu|about>`, `--open-model`, `--open-hub`, `--welcome` or `--mode <mode>`.

## Code style

- Keep code clean. Write a comment only when it explains *why*; well-named code already says *what*.
- Every user-facing string goes through `tr!("Spanish", "English")`.
- No machine-specific paths. Use settings, `dotsmith_engine::locations` or environment variables. Show paths with `settings::display_path` so the account name never appears.
- Colours and fonts come from `gpui/src/theme.rs`, and icons are Lucide SVGs registered in `gpui/src/icons.rs`.
- Engine code (`engine/`) never depends on the app. Errors there are plain English with `anyhow` context.
- Formatting is `rustfmt` with the repository's `rustfmt.toml`; lints are configured once in the root `Cargo.toml`.

## Commits and pull requests

- Write short, imperative commit subjects ("Add low-VRAM toggle"), with the reason in the body when it isn't obvious.
- Keep each pull request to one change, and add a line under `Unreleased` in `CHANGELOG.md` if it changes behaviour.
- Fill in the pull request template.

## License

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE).

## Reporting bugs

Use the bug report template, and attach `dotsmith.log` (**Settings > About > Logs > Open**).

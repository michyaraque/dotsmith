## What and why

<!-- One change per pull request. -->

## How it was checked

- [ ] `cargo test --release --lib` in `engine/`
- [ ] `cargo build --release` and `cargo test --release` in `gpui/`, no warnings
- [ ] `cargo test --release -- --ignored` (if generation changed)
- [ ] Screenshots before / after (if the UI changed)

## Checklist

- [ ] New user-facing text uses `tr!("es", "en")`
- [ ] No machine-specific paths
- [ ] `CHANGELOG.md` updated (if behaviour changed)

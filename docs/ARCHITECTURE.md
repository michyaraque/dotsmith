# Architecture

Dotsmith is two Rust crates in one process: a GPUI interface and a candle + CUDA inference engine. There is no Python, server or IPC: the UI talks to a worker thread through channels.

```
┌───────────────────────────── dotsmith.exe (gpui/) ─────────────────────────────┐
│ UI thread                                   engine thread                    │
│ studio.rs ── Backend::send(json) ──────────▶ backend.rs Worker               │
│   model_picker.rs                             ├─ jobs.rs   (modes, prompts,  │
│   settings_view.rs  ◀── poll(): Msg ──────────┤            post-processing)  │
│   creature_mode.rs      Event/Preview/Result  └─ dotsmith_engine::Engine         │
│   tools_menu.rs                                     (engine/ crate)          │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Repository layout

| Path | What it holds |
| --- | --- |
| `Cargo.toml` | The workspace (`engine`, `gpui`): shared package metadata and lints; `rust-toolchain.toml` pins the compiler |
| `gpui/` | The desktop app (binary `dotsmith.exe`) |
| `gpui/src/main.rs` | Startup: logging, settings, language, window |
| `gpui/src/studio.rs` | Main window: mode rail, controls, viewer, history, status bar |
| `gpui/src/settings.rs`, `settings_view.rs` | `%APPDATA%\Dotsmith\settings.json` and the Settings window |
| `gpui/src/backend.rs` | Worker thread that owns the engine; model switching, load timing, folders |
| `gpui/src/jobs.rs` | The generation modes: prompt building, canvases, keying, depixel, saving |
| `gpui/src/i18n.rs` | `tr!("es", "en")` and the language choice |
| `gpui/src/log.rs` | `%LOCALAPPDATA%\Dotsmith\logs\dotsmith.log` and the panic hook |
| `gpui/src/download.rs` | Resumable Hugging Face downloads on their own thread |
| `gpui/src/hf.rs` | Live model cards from Hugging Face (stats, README, sizes, avatars), cached for a day |
| `gpui/src/model_hub.rs` | The model manager and the downloads panel |
| `gpui/src/welcome.rs` | The start screen: graphics card check and every base model with its download |
| `gpui/src/model_picker.rs`, `palette.rs`, `creature_mode.rs`, `tools_menu.rs` | Sub-views of the studio: model and LoRA picker, command palette, walk sheets, tools |
| `gpui/src/aurora.rs` | The live preview: a drifting gradient of the step's colours that gives way to the image as it takes shape |
| `gpui/src/rig.rs`, `anim.rs`, `pack_split.rs`, `pixels.rs`, `clipboard.rs`, `input.rs` | Self-contained helpers |
| `gpui/scripts/` | `package.ps1` (release) and `isolated_shot.ps1` (screenshots on a hidden desktop) |
| `gpui/installer/dotsmith.iss` | Inno Setup installer |
| `engine/` | Inference library (`dotsmith_engine`) and a developer CLI (`dotsmith-engine.exe`) |
| `engine/src/engine.rs` | Generation loop: CFG, Euler steps, SDEdit, masks, previews |
| `engine/models.json` | The model catalog: files, download sources and defaults of every family (merged with `%APPDATA%\Dotsmith\models.json`) |
| `engine/src/manifest.rs` | Loads and merges the catalog |
| `engine/src/model.rs` | `DiffusionModel` trait, `Family`, `ModelKind`, `ModelPaths` |
| `engine/src/models/{qwen21,krea2,lilipixel,rdflow}/` | One folder per family: network, VAE or decoder, sampler |
| `engine/src/locations.rs` | Where the catalog's files are on this machine |
| `engine/src/gpu.rs` | CUDA device check and NVRTC target |
| `engine/src/catalog.rs` | LoRA library scan and `version.json` |
| `engine/src/sprite.rs`, `pixeloe.rs` | Pixel post-processing |
| `engine/parity/` | PyTorch references for the LiliPixel and RD Flow ports (`DOTSMITH_LILI_REF` / `DOTSMITH_RD_REF` tests) |

## Generation flow

1. The UI builds a JSON request from the form and calls `Backend::send`.
2. The worker calls `ensure_engine`: it loads the family, swaps the DiT (Base ↔ Turbo) or swaps only the LoRA, whichever is needed. The engine counts the weight bytes it reads, so the progress bar shows the real percentage and the time left; the bytes a load read are remembered for the next one.
3. `jobs::generate` adapts the request to the family and the LoRA version (scale, background key, steps), runs `Engine::generate`, and streams progress and preview textures back.
4. Each sample is keyed, depixelled and fitted, then saved as `<stamp>_<mode>_<seed>.png` plus `.json` in the output folder.
5. The UI picks up `Msg::Result`, adds it to the history and selects it.

`Esc` sets an atomic flag that the engine checks every step.

## Adding a model family

1. Add a folder under `engine/src/models/` that implements `DiffusionModel`.
2. Add the variant to `Family` (its `id`) and to `load_model`.
3. Describe it in `engine/models.json`: its Hugging Face repo, files, download sources, scale, background and capabilities.

The picker, Settings, downloads and the history read everything from the catalog, so the UI needs no changes. Changing files or sources of an existing family needs no code at all: see [MODELS.md](MODELS.md#your-own-catalog).

## GPU compatibility

- candle's kernels are built as `sm_80` PTX (`CUDA_COMPUTE_CAP = "80"`), and the driver JIT-compiles them once for the actual card. RTX 30, 40 and 50 all run the same binary.
- The engine's own fused kernels are compiled by NVRTC for the detected architecture (`gpu::nvrtc_arch`).
- The CUDA DLLs are delay-loaded (`gpui/build.rs`). Dotsmith opens on any PC, and `gpu::probe` explains what is missing before the first CUDA call.
- Low-VRAM mode is on automatically below 20 GB. It can also be forced either way in Settings.

## Conventions

- User-facing text goes through `tr!(spanish, english)`. Engine errors are plain English; the app adds context.
- No model names, files or sources in code: they belong in the catalog. Nothing Hugging Face can tell (sizes, descriptions, stats, logos) is written down at all.
- No paths of a particular machine in code. Folders come from settings, environment or `locations.rs`, and paths shown in the UI go through `settings::display_path`.
- Comments explain *why*, never *what*.

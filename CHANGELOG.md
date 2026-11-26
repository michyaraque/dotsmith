# Changelog

All notable changes to Dotsmith. The format follows [Keep a Changelog](https://keepachangelog.com) and versions follow [SemVer](https://semver.org).

## [Unreleased]

First public release of Dotsmith: a desktop studio that makes game-ready pixel art on your own NVIDIA GPU, with no account, upload or per-image cost.

### Added
- **Modes**: Sprite (16 to 192 px), Pack (2×2 or 3×3 sheets), Walls (wall sets and the closed 3×3 room), Character (8-frame walk cycles in 1 or 4 directions, with a player and GIF export), Variants, Angle (3D gizmo), Regions (repaint only the pixels you paint), Image (photo or illustration to pixel art) and Pose (a posable silhouette rig).
- **Four base models** on Dotsmith's own Rust + CUDA engine: Qwen-Image 2.1 (Base, Turbo), Krea 2 (Raw, Turbo), LiliPixel V1 (native 16, 32 and 64 px with real alpha) and RD Flow 64 (every pixel an exact palette colour). LiliPixel and RD Flow are checked against PyTorch references.
- **Runs on 6 GB cards**: when a model does not fit, its layers stream from RAM (Fast or Saver) with the same images; every model ran in under 3 GB of VRAM at sprite sizes. Low-VRAM mode, a VRAM limit and the weights' placement are in Settings > GPU & memory.
- **Start screen** that checks the graphics card and lists every base model side by side with its own download; nothing is picked for you.
- **Model manager** with each model's live Hugging Face card (publisher, stats, README, file sizes) and one-click downloads that resume, check disk space and show progress, plus any file or LoRA by link.
- **Model catalog** (`engine/models.json`, extendable from `%APPDATA%\Dotsmith\models.json`): files, sources, sizes, modes and default steps per model, with no model data in code.
- **Your own LoRAs**: any `.safetensors` anywhere in the LoRA folder, its base model read from the file, its trigger word from `version.json`, and its strength (0 to 1.5) under Advanced.
- **Live preview**: a drifting gradient of the image's colours that gives way to the real picture as it takes shape; Generate turns into Stop.
- **Real loading progress**: the percentage of weights read and the time left.
- **Steps** for every model, Turbo included, each starting from its own default.
- **History**: every result is a PNG plus its settings; Reuse settings brings them back, the history is searchable, right-click shows each result's actions, and a delete can be undone for 15 seconds.
- **Command palette** (Ctrl + K) for modes, settings, tools and folders, full keyboard navigation with Tab, and shortcuts for every main action.
- **Settings** with one setting per row: language, appearance (dark, light or Windows), folders, memory and shortcuts.
- **Tools** to split packs, build wall rooms and remove magenta backgrounds in batches.
- RAM and VRAM meters in the status bar, and a clear message for a missing driver, missing CUDA files, an older card or a full GPU.
- English and Spanish interface, following the Windows language by default.
- One instance at a time, remembered window size and mode, and a log with crash backtraces in `%LOCALAPPDATA%\Dotsmith\logs`.
- Portable zip and installer with the CUDA and MSVC runtimes included; one build runs on RTX 30, 40 and 50 cards.

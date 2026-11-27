# Changelog

All notable changes to Dotsmith. The format follows [Keep a Changelog](https://keepachangelog.com) and versions follow [SemVer](https://semver.org).

## [1.3.1] - 2026-11-27

### Fixed
- Krea 2 Turbo downloads without accepting Krea 2 Raw's license: the VAE both variants share now comes from a repo with no license gate, and every model's default files download without a Hugging Face token.
- Each variant downloads only the files it runs on.
- Testing the Hugging Face token no longer freezes the app.

### Changed
- The Hugging Face token is hidden, with a button to show it.

## [1.3.0] - 2026-11-27

Dotsmith updates itself from now on.

### Added
- **Updates from inside the app**: Dotsmith looks for a new release at start and every 12 hours, shows "Update to X" in the title bar and in Settings > About, downloads the installer for your build (standard or legacy), checks it against the release's checksums, and restarts into Dotsmith Setup, which installs it and opens Dotsmith again. The automatic check can be turned off in Settings > About; portable copies open the release page instead.

## [1.2.1] - 2026-11-26

### Fixed
- Opening `uninstall.exe` directly uninstalls Dotsmith instead of reporting a damaged installer.
- Close, Finish and the title bar's close button close the setup window.
- The setup screens use standard installer wording.

## [1.2.0] - 2026-11-26

Dotsmith installs with its own setup program, in its own look.

### Added
- **Dotsmith Setup**, an installer in Dotsmith's own look: installs for your user with no admin prompt, updates in place, replaces the install of 1.0 and 1.1, and takes about a second to copy the files. Its uninstaller sits in Windows' installed apps, keeps your images and LoRAs, and asks before deleting downloaded models or settings.

## [1.1.0] - 2026-11-26

Faster, steadier model loading on modest PCs, and fewer dead ends on the way to the first image.

### Added
- **Hugging Face page** in Settings that walks you through accepting a model's license and creating a token, and tests the token against every gated file before you download.
- **Cancel a load** from its progress screen or with Esc; the RAM and VRAM it took are freed at once.
- **New LoRAs and models show up on their own** while the app runs, once a file finishes copying, and the model picker has a refresh button to look again on demand.

### Changed
- **Loading reads ahead** with large background reads instead of one page at a time, so a cold load runs at the drive's speed: Krea 2 Turbo in Fast mode loads in 7.3 s instead of 21.9 s from an NVMe drive.
- Krea 2's tokenizer now comes from Qwen3-VL, one file less behind a Hugging Face license.

### Fixed
- On PCs with 32 GB of RAM, the automatic memory mode no longer locks the model in RAM when that leaves no room to read it, which could stall a Krea 2 load for 15 minutes.
- Failed downloads and errors open a dialog above every window instead of a line at the bottom left, and a download that needs a license says so and links to the steps.
- A model with only its Turbo variant downloaded opens and generates instead of sending you back to the start screen.

## [1.0.0] - 2026-11-26

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

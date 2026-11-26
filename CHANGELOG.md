# Changelog

All notable changes to Dotsmith. The format follows [Keep a Changelog](https://keepachangelog.com) and versions follow [SemVer](https://semver.org).

## [Unreleased]

The app is now called **Dotsmith** (it was Forja). On the first start its settings, logs, LoRA folder, downloaded models and the default results folder move from the old `Forja` folders on their own.

### Added
- RAM and VRAM meters in the status bar (VRAM counts every app on the GPU; amber from 90 %, red from 97 %).
- Image mode: from Stylisation 0.60 up, the prompt mixes into the picture and may draw past its outline (wings, weapons, effects).
- Start screen (like Obsidian's) while Dotsmith cannot generate yet: graphics card, every base model in the catalog side by side (publisher, size, its own Download or Use), "I already have the models", LoRAs and the language, then **Start creating**. No model is picked for the user. It replaces the setup card that sat inside the canvas.
- **LiliPixel V1** (Pinkstack/LiliPixel-V1-190M): a 190M pixel-space DiT that draws 16, 32 or 64 px RGBA sprites directly, with its voyage-4-nano text encoder. Ported to the Rust engine and checked against a PyTorch reference (max difference 1e-5 after 30 steps). Sprite, Variants and Regions.
- **RD Flow 64** (Retro-Diffusion/rd-flow-64): a 271M model that draws 64 x 64 pixel art where every pixel is an exact palette colour or transparent, with its own sampler (self-conditioning, palette from its own guess, pointer head) and Qwen3-VL-4B's first 27 layers as text encoder. Sprite at 64 px. Checked against a PyTorch reference: network and pointer within 1e-5, palette ramps identical, text features within bf16 rounding.
- Both download in one click from the start screen or the model manager; the studio shows only the sizes, modes and options each model supports.
- Model manager: a **Coming soon** section (catalog `store`) for models listed before Dotsmith can run them.
- Deleting a result can be undone for 15 seconds (**Undo** in the status bar, or Ctrl + Z); after that it goes to the Recycle Bin as before.
- History search: filters by description, accents ignored.
- Keyboard navigation: Tab and Shift + Tab reach every control (fields, buttons, chips, switches, model cards, menu rows) with a focus ring, Enter or Space activates it, and an open window keeps Tab inside it. A mouse click shows no ring.
- Right-click a history result for its actions: use as source, reuse settings, copy (also enlarged), show in folder, move to the Recycle Bin.
- Command palette (Ctrl + K or Ctrl + P, or the magnifier in the title bar): every mode, setting page, tool and folder in one search box, accents ignored, arrow keys and Enter.
- LoRA strength (0 to 1.5, under **Advanced** when a LoRA is selected): remembered between sessions, saved with every result and restored by Reuse.
- Steps can be changed for every model, Turbo included (default 8). Each base starts from its own default, set per variant in the catalog (`variants.*.steps`).
- Model loading shows real progress, like LM Studio: the percentage of weight bytes read and the time left at the current rate, instead of a guess from the previous load's duration.

### Changed
- New icon: a gold pixel-art "D" holding one glowing pixel, drawn with Krea 2 Turbo.
- Settings > About credits the author, Michael Araque, with links to GitHub and X.
- The README states the VRAM and RAM each model really needs, measured in every memory mode: with the layers in RAM every model runs in under 3 GB of VRAM at sprite sizes, so any RTX 30 card with 6 GB works.
- Engine and app build as one Cargo workspace (one lockfile, one `target/`, the toolchain pinned in `rust-toolchain.toml`), formatted with `rustfmt` and checked with clippy at zero warnings.
- The engine CLI (`dotsmith-engine`) keeps `generate`, `batch`, `mem-cycle`, `pixelize` and `post`; the one-off parity and experiment commands are gone.
- Dotsmith no longer expects a project layout. Results go to one output folder (`Pictures\Dotsmith` by default) instead of next to each LoRA; the LoRA folder defaults to `%LOCALAPPDATA%\Dotsmith\loras` and any `.safetensors` inside it is listed, its base model recognised from the file itself. A LoRA's trigger word comes from its `version.json` / `.dotsmith.json` (`trigger`) instead of being built in.
- Settings rebuilt as one setting per row with the control on the right (General, Models & LoRAs, GPU & memory, Shortcuts, About); memory options use segmented controls.
- The title bar groups Tools, Downloads and Settings as icon buttons on the right, and the model selector is a fixed-width bar on the window's centre line (Ctrl + L opens it); the status bar is one strip with small RAM / VRAM usage bars.
- The result card is one line of facts plus the model, with Reuse and Use as source as the only text buttons.
- Side panel: explanations moved into tooltips, seed / steps / prompt strength under a collapsible **Advanced** row with a one-line summary, and Generate turns into Stop (with progress) while running.
- Model picker: **No LoRA** is a choice in the list, an empty library offers to open the LoRA folder, and the footer hints are plain language.
- Modals share one backdrop and fade in.
- Zoom and pan in the canvas stay put when another result is selected, so results can be compared at the same spot.
- The live preview is a soft, slowly drifting gradient of the image's own colours (like a mesh gradient): each step blends into the next, and as soon as the sprite takes shape the field gives way to the real preview (a LoRA's flat backdrop is keyed out, so the sprite emerges over the colours instead of a magenta flash).
- Image mode is one flow: pick the sprite size (an exact square of 32–128 px, the picture fitted inside without distortion) and optionally redraw with the LoRA. The background works like the Sprite tab: "Opaque background" keeps the whole picture; off, a flat background is removed and the image cropped to the object (a background that is not flat is kept, and the result says so). The old fixed 32 × 32 "single object" path, which invented items unrelated to the picture, is gone.

### Fixed
- Forcing the weights into RAM (or a small card in Auto) failed to load RD Flow 64: a layer stack with an odd number of layers wrapped its resident count around. It now keeps one more layer on the GPU.
- Clicking below the last line of a text field put the cursor at the start; it now goes to the end.
- Image mode: noisy flat backgrounds (resampled screenshots, a frame around a magenta backdrop) are now removed with their coloured rim instead of being kept, and small sources (an existing 64 px sprite) are accepted instead of rejected.
- Image mode: a crash (division by zero) when the source was less than 4x the output size.
- img2img on canvases over 512 px encodes the starting image in tiles, so a 128 px img2sprite no longer runs out of VRAM.
- A generation that does not fit in VRAM (for example a 3x3 · 64 px pack while other apps hold GPU memory) no longer freezes: Windows silently moves the overflow to system RAM, so the engine now watches free VRAM and the process's Windows VRAM budget, and stops within one DiT, VAE or text-encoder block with a clear message. Stop answers just as fast, also while decoding.
- Krea 2 decodes big canvases in 512 px tiles: a 1536 px pack peaks at 17.6 GB instead of 20.4 GB.

## [1.0.0] - 2026-11-23

First release that runs on any Windows PC with an RTX GPU, not only on the development machine.

### Added
- **Settings** window (gear in the title bar, Ctrl + ,): language, output folder, base-model folder, Hugging Face folders, LoRA library, low-VRAM mode, GPU info, logs and licenses.
- Per-file model check: every file of every family is shown as found or missing, with the folder where Dotsmith expects it.
- Model manager (from the model selector): every family with its live Hugging Face card (publisher avatar, downloads, likes, last update, licence, parameters, description and README), download options per variant with real file sizes, and the state of every file.
- Downloads panel in the title bar: the running batch file by file, and what finished in this session.
- Downloads from Hugging Face: missing model files with one click, or any file or LoRA by link. Resumable, cancellable, with a disk-space check and token support for gated repos.
- Model catalog (`engine/models.json`, extendable with `%APPDATA%\Dotsmith\models.json`): which files each family needs and where they come from is data, not code; everything Hugging Face knows is read from it.
- Krea 2 can be downloaded completely: the engine also reads ComfyUI's Qwen3-VL text encoder layout.
- First-run card that explains what is missing (GPU or model files) and opens the right Settings tab.
- English interface alongside Spanish. It follows the Windows display language by default.
- GPU check before any load: no driver, missing CUDA DLLs or a pre-RTX 30 card each get a clear message.
- Friendly out-of-memory message that points to low-VRAM mode.
- Keyboard: ← / → step through results, Delete moves the result to the Recycle Bin.
- Window size, maximised state and last mode are remembered.
- One Dotsmith per session: opening it again brings the running window forward instead of loading a second model onto the GPU.
- The history loads on a background thread, so switching models never freezes the window.
- Log file with crash backtraces in `%LOCALAPPDATA%\Dotsmith\logs`.
- Release packaging: portable zip with the CUDA and MSVC runtimes, and an Inno Setup installer.
- Exe version information and product metadata.
- MIT license.

### Changed
- Model files are found through settings, environment variables or `%LOCALAPPDATA%\Dotsmith\models`. Paths of the development machine were removed.
- The LoRA library can be any folder with `qwen21` / `krea2` subfolders, and Dotsmith starts without one.
- One build runs on RTX 30, 40 and 50 cards: candle kernels ship as sm_80 PTX, and the engine's own kernels compile for the detected GPU.
- Generating without a LoRA follows the family's scale and background (Krea 2 was treated as x16).
- Results without a LoRA go to `Pictures\Dotsmith`. Any output folder can be chosen.
- Delete moves results to the Recycle Bin instead of erasing them.
- The result card's actions fit on one row; folder and delete are icon buttons with tooltips.
- Navigation: the model selector moved to the centre of the title bar; the mode rail is split into "Create" and "Edit an image" with tooltips; the status bar only shows state, and its output folder opens on click.
- Leaner interface text: no subtitle under every mode, no hints that repeat their label, no internal notes.
- Secondary text meets WCAG AA contrast (4.5:1).
- Recent models show the LoRA first, with the full name in a tooltip.
- The executable is now `dotsmith.exe`.

## [0.1.0]
- Native GPUI studio with the Rust engine: Sprite, Pack, Walls, Character, Variants, Angle, Regions, Image and Pose modes.


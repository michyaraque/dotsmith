# Models

Dotsmith runs model *families*: an architecture the engine can execute. What Dotsmith needs to know about a family lives in a catalog, [`engine/models.json`](../engine/models.json): which files it needs, where to look for them, where to download them, and how its output is post-processed. Everything Hugging Face already knows is read live from the family's repo instead:

- publisher and avatar
- description and README
- downloads, likes and last update
- licence, task and parameter count
- whether the repo is gated, and the size of every file

Live information is cached in `%LOCALAPPDATA%\Dotsmith\cache\hf` for a day, so it still shows offline.

| Family | Repo | Downloads |
| --- | --- | --- |
| Qwen-Image 2.1 (`qwen21`) | `Qwen/Qwen-Image-2.1` | Every file, from `Comfy-Org/Qwen-Image-2.1` and `Qwen/Qwen-Image-2.1` |
| Krea 2 (`krea2`) | `krea/Krea-2-Raw` | DiT and text encoder from `Comfy-Org/Krea-2`; tokenizer and VAE from `krea/Krea-2-Raw` (gated) |

## The model manager

The model selector in the title bar has a **Manage models** button that opens the manager; so does the first-run card. On the left are the families; on the right is the selected family's card, with:

- its stats, description, specs and capabilities
- one download option per variant (base, turbo), with the size still missing
- every file, with ✓ or ✗ and its size
- the README from Hugging Face

Downloads go to the models folder (`%LOCALAPPDATA%\Dotsmith\models` unless you chose another). They resume after an interruption and check free disk space first. The download icon in the title bar shows the running batch file by file, plus what finished in this session.

Gated repos ask you to accept their licence on huggingface.co first. Paste it in **Settings > Models & LoRAs > Hugging Face > Access token**; the `HF_TOKEN` environment variable also works.

**Settings > Models & LoRAs > Hugging Face > Download a file** fetches any file: type `owner/repo/path/to/file` or paste a link. **To base models** keeps the repo's path. **As a LoRA** saves it to `<library>/<family>/downloads/<repo>/`.

## Where Dotsmith looks

For every catalog entry, in order:

1. The file named by the entry's `env` variable, if it has one
2. `<models folder>/<file>`, for each models folder: Settings, `DOTSMITH_MODELS`, `AI_TOOLKIT\models`, `%LOCALAPPDATA%\Dotsmith\models`, `models` next to `dotsmith.exe`
3. The entry's `also` patterns:

| Pattern | Means |
| --- | --- |
| `{hf:Name}/rest` | `rest` inside every Hugging Face folder added in Settings, and inside `<models folder>/Name` or `<models folder>/hf/Name` |
| `{hub:owner/repo}/rest` | `rest` inside the local Hugging Face cache snapshots of that repo |
| `{env:VAR}/rest` | `rest` inside the folder named by `VAR` |

## Your own catalog

Create `%APPDATA%\Dotsmith\models.json` and Dotsmith merges it over the built-in one. Objects merge key by key; lists and values replace the built-in ones. A broken file is ignored, and Dotsmith shows the error.

For example, to point Krea 2 at your own published conversions and a fine-tuned DiT:

```json
{
  "families": {
    "krea2": {
      "files": [
        { "role": "diffusion model", "file": "diffusion_models/my_krea2_finetune_int8.safetensors",
          "source": { "repo": "you/my-krea2", "path": "my_krea2_finetune_int8.safetensors" } },
        { "role": "diffusion model", "variant": "turbo", "file": "diffusion_models/krea2_turbo_int8_convrot.safetensors",
          "source": { "repo": "you/krea2-int8", "path": "krea2_turbo_int8_convrot.safetensors" } },
        { "role": "text encoder", "file": "text_encoders/krea2_qwen3vl_4b_int8_convrot.safetensors",
          "source": { "repo": "you/krea2-int8", "path": "krea2_qwen3vl_4b_int8_convrot.safetensors" } },
        { "role": "VAE", "file": "vae/krea2_vae.safetensors",
          "source": { "repo": "krea/Krea-2-Raw", "path": "vae/diffusion_pytorch_model.safetensors" } },
        { "role": "tokenizer", "file": "tokenizers/krea2/tokenizer.json",
          "source": { "repo": "krea/Krea-2-Raw", "path": "tokenizer/tokenizer.json" } }
      ]
    }
  }
}
```

### Family fields

| Field | Meaning |
| --- | --- |
| `label` | Name in the interface |
| `repo` | Hugging Face repo whose card describes the family |
| `variants.base` / `variants.turbo` | `name` (button text), `label` (full model name, also stored with every result) and `steps` (default sampling steps; the user can change them) |
| `scale` | Image pixels per sprite pixel without a LoRA |
| `background` | `"#RRGGBB"` key colour, or `null` for real transparency |
| `negative_prompt` | Whether the negative prompt is sent |
| `fixed_guidance` | Whether guidance and Turbo steps always follow the family defaults |
| `capabilities` | Extra modes the family supports; `walk_sheet` enables Character |
| `lora_modules` | Module names only this architecture has (`N` = block number): how a loose LoRA file is matched to the family |
| `prompt` | `"plain"`: the model reads the description as written (no trigger, layout words or size flags, no clean-up after) |
| `sizes` | Square sizes the model draws; the studio offers only these |
| `modes` | Studio modes it supports (`single`, `variants`, ...); the others are dimmed |
| `opaque` | Whether "Opaque background" means something for it (default true) |
| `files` | `role` (`diffusion model`, `text encoder`, `VAE`, `tokenizer`), `variant`, `file`, `also`, `env`, `source { repo, path }` |

A family without a few-step model leaves `variants.turbo.name` empty: no Turbo switch is shown.

Next to `families`, a `store` list names models that are not runnable yet (`{"repo": "owner/name", "label": "Name"}`): the model manager shows them under **Coming soon** with their live Hugging Face card and no download.

A new architecture still needs code: a folder under `engine/src/models/` that implements `DiffusionModel`, plus an entry in `Family` and `load_model`. See [ARCHITECTURE.md](ARCHITECTURE.md).

## LoRA library

The LoRA folder is `%LOCALAPPDATA%\Dotsmith\loras` unless you pick another one in **Settings > Models & LoRAs**. Drop any `.safetensors` LoRA anywhere inside it: Dotsmith reads each file's tensor names to know which base model it belongs to (the catalog's `lora_modules`) and lists the files of a folder together. Training output keeps its own grouping, with checkpoints by step:

```
<library>/
  any/folder/pixel_icons.safetensors                         listed under its file name
  <family id>/<version>/<run>/<run>.safetensors              "final"
  <family id>/<version>/<run>/<run>_000006500.safetensors    step 6500
```

A `version.json` in a folder describes how its LoRAs were trained and overrides the family's defaults; a `<file>.dotsmith.json` next to one file overrides both:

```json
{ "base": "krea2", "scale": 8, "background": "#FF00FF", "trigger": "mystyle" }
```

`trigger` is the word the LoRA was trained on; Dotsmith starts every prompt with it.

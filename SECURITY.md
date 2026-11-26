# Security

## Supported versions

Only the latest release gets fixes.

## What Dotsmith does and doesn't do

- Dotsmith makes no network connections. Generation, settings and logs stay on your PC.
- It reads model weights (`.safetensors`, `.gguf`) and images you choose, and writes PNG / JSON / GIF files to the output folder.
- Settings live in `%APPDATA%\Dotsmith`, and logs in `%LOCALAPPDATA%\Dotsmith\logs`. Logs contain file paths and error messages, never prompts or images.
- Deleting a result moves it to the Recycle Bin.

Only load model files from sources you trust. `.safetensors` can't run code, but a crafted file could still crash the app.

## Reporting a vulnerability

Please don't open a public issue. Use GitHub's **Report a vulnerability** (Security > Advisories) on this repository and include:

- the Dotsmith version (**Settings > About**),
- steps or a file that reproduces it,
- what you expected and what happened.

You will get an answer within 7 days.

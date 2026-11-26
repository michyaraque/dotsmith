# Builds Dotsmith and assembles a self-contained Windows release in dist\ at the repository root:
#   dist\Dotsmith\                         dotsmith.exe + CUDA runtime + VC runtime + docs
#   dist\Dotsmith-<ver><suffix>-win64.zip  portable build
#   dist\Dotsmith-<ver><suffix>-setup.exe  installer (installer\: Dotsmith Setup with the portable folder packed behind it)
# The CUDA runtime is taken from CUDA_PATH, the toolkit the build used; CARGO_TARGET_DIR picks a separate build.
param([switch]$SkipBuild, [string]$Suffix = '')
$ErrorActionPreference = 'Stop'

$root = Split-Path $PSScriptRoot -Parent
$repo = Split-Path $root -Parent
$version = (Select-String -Path "$root\Cargo.toml" -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $repo 'target' }
$dist = Join-Path $repo 'dist'
$out = Join-Path $dist 'Dotsmith'

if (-not $SkipBuild) {
    Push-Location $repo
    cargo build --release -p dotsmith -p dotsmith-setup
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'cargo build failed' }
    Pop-Location
}

if (-not $env:CUDA_PATH) { throw 'CUDA_PATH is not set (the CUDA toolkit the build used, to collect its runtime DLLs)' }
$cudaBin = Join-Path $env:CUDA_PATH 'bin\x64'
if (-not (Test-Path (Join-Path $cudaBin 'cublas64_*.dll'))) { $cudaBin = Join-Path $env:CUDA_PATH 'bin' }
$major = [int]([regex]::Match((& (Join-Path $env:CUDA_PATH 'bin\nvcc.exe') --version | Out-String), 'release (\d+)\.').Groups[1].Value)
$cudaDlls = @("cublas64_$major.dll", "cublasLt64_$major.dll", 'curand64_10.dll', "nvrtc64_${major}0_0.dll")
$cudaDlls += (Get-ChildItem $cudaBin -Filter 'nvrtc-builtins64_*.dll' | Select-Object -First 1).Name
# CUDA 12 links candle's static kernel library against the runtime DLL
if ($major -lt 13) { $cudaDlls += "cudart64_$major.dll" }
$eula = @('EULA.txt', 'LICENSE') | ForEach-Object { Join-Path $env:CUDA_PATH $_ } | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $eula) { throw "no NVIDIA license (EULA.txt or LICENSE) in $env:CUDA_PATH" }

$vsRedist = Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC143.CRT\vcruntime140.dll' -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | Select-Object -First 1
if (-not $vsRedist) { throw 'vcruntime140.dll not found in the Visual Studio redistributables' }

if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null

Copy-Item (Join-Path $target 'release\dotsmith.exe') $out
foreach ($d in $cudaDlls) { Copy-Item (Join-Path $cudaBin $d) $out }
Copy-Item $vsRedist.FullName $out
Copy-Item "$repo\README.md", "$repo\README.es.md", "$repo\CHANGELOG.md", "$repo\LICENSE", "$repo\docs\MODELS.md" $out
New-Item -ItemType Directory -Force "$out\licenses" | Out-Null
Copy-Item "$root\assets\icons\LICENSE" "$out\licenses\lucide-ISC.txt"
Copy-Item $eula "$out\licenses\NVIDIA-CUDA-EULA.txt"

$zip = Join-Path $dist "Dotsmith-$version$Suffix-win64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $out -DestinationPath $zip -CompressionLevel Optimal
Write-Host "portable: $zip"

$setup = Join-Path $dist "Dotsmith-$version$Suffix-setup.exe"
$err = Join-Path $dist 'setup-pack.log'
# a windowed program: Start-Process waits for it and gives its exit code
$p = Start-Process -FilePath (Join-Path $target 'release\dotsmith-setup.exe') -ArgumentList '--pack', "`"$out`"", "`"$setup`"" -Wait -PassThru -NoNewWindow -RedirectStandardError $err
if ($p.ExitCode -ne 0) { throw "installer build failed: $(Get-Content $err -Raw)" }
Remove-Item $err
Write-Host "installer: $setup"

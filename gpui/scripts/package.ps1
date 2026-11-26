# Builds Dotsmith and assembles a self-contained Windows release in dist\ at the repository root:
#   dist\Dotsmith\                 dotsmith.exe + CUDA runtime + VC runtime + docs
#   dist\Dotsmith-<ver>-win64.zip  portable build
#   dist\Dotsmith-<ver>-setup.exe  installer (only when Inno Setup 6 is installed)
param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'

$root = Split-Path $PSScriptRoot -Parent
$repo = Split-Path $root -Parent
$version = (Select-String -Path "$root\Cargo.toml" -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$dist = Join-Path $repo 'dist'
$out = Join-Path $dist 'Dotsmith'

if (-not $SkipBuild) {
    Push-Location $repo
    cargo build --release -p dotsmith
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'cargo build failed' }
    Pop-Location
}

if (-not $env:CUDA_PATH) { throw 'CUDA_PATH is not set (CUDA 13 toolkit required to collect the runtime DLLs)' }
$cudaBin = Join-Path $env:CUDA_PATH 'bin\x64'
if (-not (Test-Path $cudaBin)) { $cudaBin = Join-Path $env:CUDA_PATH 'bin' }
$cudaDlls = 'cublas64_13.dll', 'cublasLt64_13.dll', 'curand64_10.dll', 'nvrtc64_130_0.dll', 'nvrtc-builtins64_131.dll'

$vsRedist = Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC143.CRT\vcruntime140.dll' -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | Select-Object -First 1
if (-not $vsRedist) { throw 'vcruntime140.dll not found in the Visual Studio redistributables' }

if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null

Copy-Item "$repo\target\release\dotsmith.exe" $out
foreach ($d in $cudaDlls) { Copy-Item (Join-Path $cudaBin $d) $out }
Copy-Item $vsRedist.FullName $out
Copy-Item "$repo\README.md", "$repo\README.es.md", "$repo\CHANGELOG.md", "$repo\LICENSE", "$repo\docs\MODELS.md" $out
New-Item -ItemType Directory -Force "$out\licenses" | Out-Null
Copy-Item "$root\assets\icons\LICENSE" "$out\licenses\lucide-ISC.txt"
Copy-Item (Join-Path $env:CUDA_PATH 'EULA.txt') "$out\licenses\NVIDIA-CUDA-EULA.txt"

$zip = Join-Path $dist "Dotsmith-$version-win64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $out -DestinationPath $zip -CompressionLevel Optimal
Write-Host "portable: $zip"

$iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($iscc) {
    & $iscc "/DAppVersion=$version" "/DSourceDir=$out" "/O$dist" "$root\installer\dotsmith.iss"
    if ($LASTEXITCODE -ne 0) { throw 'installer build failed' }
    Write-Host "installer: $dist\Dotsmith-$version-setup.exe"
} else {
    Write-Host 'Inno Setup 6 not found: skipped the installer (https://jrsoftware.org/isinfo.php)'
}

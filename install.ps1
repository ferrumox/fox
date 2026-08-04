# install.ps1 — Install fox (ferrumox) on Windows from GitHub Releases.
#
# Usage (run in PowerShell as Administrator or with user-writable install dir):
#   irm https://github.com/ferrumox/fox/releases/latest/download/install.ps1 | iex
#
# Options (set before piping or pass as env vars):
#   $env:FOX_VERSION = "v1.0.0"          # specific version (default: latest)
#   $env:FOX_INSTALL_DIR = "C:\fox"      # install directory (default: %LOCALAPPDATA%\ferrumox\bin)

param(
    [string]$Version  = $env:FOX_VERSION,
    [string]$InstallDir = $env:FOX_INSTALL_DIR
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Repo = "ferrumox/fox"
$Target = "x86_64-pc-windows-msvc"

if (-not $InstallDir) {
    $InstallDir = Join-Path $env:LOCALAPPDATA "ferrumox\bin"
}

# Resolve latest version if not specified.
if (-not $Version) {
    Write-Host "Fetching latest version..."
    $release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
    $Version = $release.tag_name
}

$VersionNum = $Version.TrimStart('v')
$ZipName    = "fox-${VersionNum}-${Target}.zip"
$Url        = "https://github.com/$Repo/releases/download/$Version/$ZipName"

# Check the asset exists before promising anything. fox's release workflow currently
# builds Linux x86_64 only — the Windows target was removed pending verification — so
# this script could never succeed, and said so with a bare 404 from Invoke-WebRequest.
# Probing the release means the message is accurate today and this script starts working
# by itself the day a Windows build is published.
try {
    $rel = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/tags/$Version"
    $names = @($rel.assets | ForEach-Object { $_.name })
} catch { $names = @() }
if ($names.Count -gt 0 -and ($names -notcontains $ZipName)) {
    Write-Host ""
    Write-Host "No Windows build in release $Version." -ForegroundColor Red
    Write-Host "fox publishes Linux x86_64 binaries only right now. On Windows:"
    Write-Host ""
    Write-Host "  - WSL2, then the Linux installer:"
    Write-Host "      curl -fsSL https://github.com/$Repo/releases/latest/download/install.sh | sh"
    Write-Host "  - or build it: git clone --recurse-submodules https://github.com/$Repo"
    Write-Host "                 cd fox; cargo build --release --bin fox"
    Write-Host ""
    Write-Host "That release published:" ($names -join ", ")
    exit 1
}
$TmpDir     = Join-Path $env:TEMP "fox-install-$([System.IO.Path]::GetRandomFileName())"

Write-Host "Installing fox $Version..."
Write-Host "Downloading $Url"

try {
    New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
    $ZipPath = Join-Path $TmpDir $ZipName

    Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing

    Expand-Archive -Path $ZipPath -DestinationPath $TmpDir -Force

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Copy-Item (Join-Path $TmpDir "fox.exe") (Join-Path $InstallDir "fox.exe") -Force

    Write-Host ""
    Write-Host "Installed fox.exe to $InstallDir"

    # Add to PATH for current session.
    $env:PATH = "$InstallDir;$env:PATH"

    # Offer to add to user PATH permanently.
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$InstallDir*") {
        Write-Host ""
        Write-Host "To add fox to your PATH permanently, run:"
        Write-Host "  [Environment]::SetEnvironmentVariable('PATH', '$InstallDir;' + [Environment]::GetEnvironmentVariable('PATH','User'), 'User')"
    }

    Write-Host ""
    Write-Host "Run: fox --help"
}
finally {
    Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue
}

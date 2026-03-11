# install.ps1 — Install fox (ferrumox) on Windows from GitHub Releases.
#
# Usage (run in PowerShell as Administrator or with user-writable install dir):
#   irm https://raw.githubusercontent.com/your-org/ferrumox/main/install.ps1 | iex
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

$Repo = "your-org/ferrumox"
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

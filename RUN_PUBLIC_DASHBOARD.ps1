$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:FALL_STATIC_WEB = "1"

function Install-NodeDependenciesIfNeeded {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectDir,

        [Parameter(Mandatory = $true)]
        [string]$RequiredTool
    )

    $toolPath = Join-Path $ProjectDir "node_modules\.bin\$RequiredTool.cmd"
    if (Test-Path -LiteralPath $toolPath) {
        return
    }

    Write-Host "[setup] Missing $RequiredTool. Installing Node dependencies..."
    Push-Location -LiteralPath $ProjectDir
    try {
        $installExit = 0
        if (Test-Path -LiteralPath (Join-Path $ProjectDir "package-lock.json")) {
            & npm.cmd ci --no-audit --no-fund
            $installExit = $LASTEXITCODE
            if ($installExit -ne 0) {
                Write-Host "[setup] npm ci failed; retrying npm install..."
                & npm.cmd install --no-audit --no-fund
                $installExit = $LASTEXITCODE
            }
        } else {
            & npm.cmd install --no-audit --no-fund
            $installExit = $LASTEXITCODE
        }

        if ($installExit -ne 0) {
            throw "Node dependency install failed with code $installExit."
        }
    } finally {
        Pop-Location
    }
}

Write-Host "[1/3] Building web dashboard..."
$frontend = Get-ChildItem -LiteralPath $root -Directory -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -eq "frontend" -and
        (Test-Path -LiteralPath (Join-Path $_.FullName "package.json"))
    } |
    Select-Object -First 1

if (-not $frontend) {
    throw "Could not find the web dashboard frontend folder."
}

Install-NodeDependenciesIfNeeded -ProjectDir $frontend.FullName -RequiredTool "vite"
Push-Location -LiteralPath $frontend.FullName
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) {
        throw "Web dashboard build failed with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "[2/3] Starting desktop app and local web server..."
$launcherCommand = "cd /d `"$root`" && set `"PYTHONUTF8=1`" && set `"PYTHONIOENCODING=utf-8`" && set `"FALL_STATIC_WEB=1`" && python launcher.py"
Start-Process -FilePath "cmd.exe" -ArgumentList @("/k", $launcherCommand) -WorkingDirectory $root

Write-Host "Waiting for backend on http://127.0.0.1:8000 ..."
$ready = $false
$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $ready) {
    throw "The backend did not answer on http://127.0.0.1:8000. Tunnel was not started."
}

Write-Host ""
Write-Host "[3/3] Starting Cloudflare Tunnel..."
Write-Host "Copy the printed https://*.trycloudflare.com URL into the Android app."
Write-Host ""

$cloudflared = Join-Path $root "tools\cloudflared.exe"
if (Test-Path -LiteralPath $cloudflared) {
    & $cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
} else {
    & cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
}

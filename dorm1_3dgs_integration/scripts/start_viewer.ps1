param([int]$Port = 8766)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$panoramaDir = Join-Path $root 'outputs\render_4k_sharpened'
if (-not (Test-Path (Join-Path $panoramaDir 'dorm1-integration.tour.json'))) {
    $panoramaDir = Join-Path $root 'outputs\render_4k'
}
$tour = Join-Path $panoramaDir 'dorm1-integration.tour.json'
if (-not (Test-Path -LiteralPath $tour)) {
    throw '먼저 scripts\render_all_keyframes.ps1을 실행하십시오.'
}
Start-Process "http://127.0.0.1:$Port/?tour=default&integration=1"
& $python (Join-Path $root 'viewer\server.py') --host 127.0.0.1 --port $Port --panorama-dir $panoramaDir --tour $tour

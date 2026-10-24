param(
    [Parameter(Mandatory = $true)][string]$CameraJson,
    [string]$OutputDir = '',
    [int]$FaceSize = 1024,
    [int]$OutputWidth = 4096
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'enter_environment.ps1')
if (-not $OutputDir) { $OutputDir = Join-Path $root 'outputs\custom_camera' }
$run = Join-Path $root 'model\dorm1_nerfstudio_refined_1024\splatfacto\2026-10-19_221340'
& $IntegrationPython (Join-Path $root 'src\render_360_keyframes.py') `
    --load-config (Join-Path $run 'config.yml') `
    --checkpoint-root (Join-Path $root 'model') `
    --data-dir (Join-Path $root 'inference_data') `
    --camera-json (Resolve-Path -LiteralPath $CameraJson).Path `
    --output-dir $OutputDir `
    --face-size $FaceSize `
    --face-fov-deg 100 `
    --output-width $OutputWidth
if ($LASTEXITCODE -ne 0) { throw "3DGS rendering failed: $LASTEXITCODE" }

param(
    [int]$FaceSize = 1024,
    [int]$OutputWidth = 4096,
    [switch]$SkipReferenceSharpening
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'enter_environment.ps1')
$run = Join-Path $root 'model\dorm1_nerfstudio_refined_1024\splatfacto\2026-10-19_221340'
$rawOutput = Join-Path $root 'outputs\render_4k'
$cameraFiles = Get-ChildItem (Join-Path $root 'cameras\keyframes\map_keyframe_*_camera.json') | Sort-Object Name | ForEach-Object { $_.FullName }

& $IntegrationPython (Join-Path $root 'src\render_360_keyframes.py') `
    --load-config (Join-Path $run 'config.yml') `
    --checkpoint-root (Join-Path $root 'model') `
    --data-dir (Join-Path $root 'inference_data') `
    --camera-json $cameraFiles `
    --output-dir $rawOutput `
    --face-size $FaceSize `
    --face-fov-deg 100 `
    --output-width $OutputWidth
if ($LASTEXITCODE -ne 0) { throw "3DGS rendering failed: $LASTEXITCODE" }

$tourInput = Join-Path $rawOutput 'render_summary.json'
$tourOutputDir = $rawOutput
if (-not $SkipReferenceSharpening) {
    $sharpOutput = Join-Path $root 'outputs\render_4k_sharpened'
    & $IntegrationPython (Join-Path $root 'src\reference_sharpen_erp.py') `
        --render-summary $tourInput `
        --original-cameras (Join-Path $root 'cameras\cameras.npz') `
        --frames-dir (Join-Path $root 'reference_frames') `
        --output-dir $sharpOutput `
        --strength 0.65 `
        --detail-limit 24 `
        --agreement-sigma 32
    if ($LASTEXITCODE -ne 0) { throw "Reference sharpening failed: $LASTEXITCODE" }
    $tourInput = Join-Path $sharpOutput 'render_summary.json'
    $tourOutputDir = $sharpOutput
}

& $IntegrationPython (Join-Path $root 'src\build_autotour_manifest.py') `
    --render-summary $tourInput `
    --output (Join-Path $tourOutputDir 'dorm1-integration.tour.json') `
    --title 'Dorm1 · 3DGS 통합 결과' `
    --link-mode sequence
if ($LASTEXITCODE -ne 0) { throw "Tour manifest generation failed: $LASTEXITCODE" }
Write-Host "Complete: $tourOutputDir"

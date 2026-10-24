param(
    [string]$Data = (Join-Path $PSScriptRoot '..\outputs\dorm1_nerfstudio_refined_1024'),
    [string]$OutputDir = (Join-Path $PSScriptRoot '..\outputs\nerfstudio_runs_refined'),
    [int]$MaxIterations = 60000
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:VSLANG = '1033'
$env:MAX_JOBS = '4'
$env:TORCH_CUDA_ARCH_LIST = '8.9'
$startupHooks = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\tools\python_startup')).Path
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$startupHooks;$env:PYTHONPATH" } else { $startupHooks }

$cudaHome = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_HOME = $cudaHome
$env:CUDA_PATH = $cudaHome
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$developerEnvironment = & cmd.exe /d /c "call `"$vcvars`" >nul && set"
foreach ($line in $developerEnvironment) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

$venvScripts = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\.venv-nerfstudio\Scripts')).Path
$env:PATH = "$venvScripts;$(Join-Path $cudaHome 'bin');$env:PATH"
$trainer = Join-Path $venvScripts 'ns-train.exe'

& $trainer splatfacto `
    --data (Resolve-Path -LiteralPath $Data).Path `
    --output-dir $OutputDir `
    --max-num-iterations $MaxIterations `
    --vis tensorboard `
    --pipeline.datamanager.cache-images cpu `
    --pipeline.model.use-bilateral-grid True `
    --pipeline.model.densify-grad-thresh 0.0004 `
    --pipeline.model.stop-split-at 30000 `
    --pipeline.model.camera-optimizer.mode off `
    nerfstudio-data `
    --data (Resolve-Path -LiteralPath $Data).Path `
    --orientation-method none `
    --center-method none `
    --auto-scale-poses False `
    --load-3D-points True

if ($LASTEXITCODE -ne 0) {
    throw "Refined 1024 Splatfacto training failed with exit code $LASTEXITCODE"
}

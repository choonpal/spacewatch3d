param(
    [string]$Environment = (Join-Path (Split-Path -Parent $PSScriptRoot) '.venv')
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'
$cudaHome = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
if (-not (Test-Path -LiteralPath (Join-Path $cudaHome 'bin\nvcc.exe'))) {
    throw 'CUDA Toolkit 12.8이 필요합니다.'
}
if (-not (Test-Path -LiteralPath $Environment)) {
    python -m venv --system-site-packages $Environment
}
$python = Join-Path $Environment 'Scripts\python.exe'
& $python -m pip install --upgrade pip 'setuptools<82'
& $python -m pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' --index-url 'https://download.pytorch.org/whl/cu128'
& $python -m pip install --no-deps 'nerfstudio==1.1.5'
& $python -m pip install (Join-Path $root 'tools\fpsample_fallback')
& $python -m pip install 'nerfstudio==1.1.5' ninja 'numpy==2.2.6' 'fsspec==2026.2.0'
& $python -c "import torch, nerfstudio, gsplat, fpsample; assert torch.cuda.is_available(); print('Ready:', torch.cuda.get_device_name(0))"

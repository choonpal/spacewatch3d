$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:VSLANG = '1033'
$env:MAX_JOBS = '4'
$env:TORCH_CUDA_ARCH_LIST = '8.9'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = '1'
$env:PYTHONPATH = "$(Join-Path $root 'tools\python_startup');$env:PYTHONPATH"
$cudaHome = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_HOME = $cudaHome
$env:CUDA_PATH = $cudaHome
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $vcvars)) {
    throw 'Visual Studio 2022 Build Tools C++ workload가 필요합니다.'
}
$developerEnvironment = & cmd.exe /d /c "call `"$vcvars`" >nul && set"
foreach ($line in $developerEnvironment) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
$script:IntegrationPython = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $script:IntegrationPython)) {
    throw '먼저 scripts\setup_environment.ps1을 실행하십시오.'
}
$env:PATH = "$(Split-Path -Parent $script:IntegrationPython);$(Join-Path $cudaHome 'bin');$env:PATH"

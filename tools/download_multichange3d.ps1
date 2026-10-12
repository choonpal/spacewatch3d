# Official real Open Space RGB-D pair_01; anonymous public share, no account.
$ErrorActionPreference = 'Stop'
$datasetDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'data/change_evidence/multichange3d_open_space/full'
New-Item -ItemType Directory -Force -Path $datasetDir | Out-Null
$shareUrl = 'https://fbk.sharepoint.com/:f:/s/BENCHMARKS/IgCzHpZzgXjDQI1O1DeZniyZAatqJWchrICnsM1KF7eV2og?e=NCmHOp'
$changeDataSession = $null
$downloadIds = @('dabb523a-ca5b-43fb-9ced-b7a0a5e9f90e', '13d765f5-45dc-484f-9ca6-8f60c8b6ac3b')
$expectedSizes = @(73061170, 57398995)
$expectedHashes = @('85120090f2da10e70042b1575676b6fb2655f547479cc4e120b29c5592983124', '86d83d88811a447e37422e85ef6eb6aa54bcf975da5544546ff74e5cb06faaf2')
for ($epoch = 0; $epoch -lt 2; $epoch++) {
    $targetFile = Join-Path $datasetDir "$epoch.ply"
    if (!(Test-Path -LiteralPath $targetFile) -or (Get-Item -LiteralPath $targetFile).Length -ne $expectedSizes[$epoch]) {
        if ($null -eq $changeDataSession) {
            Invoke-WebRequest -Uri $shareUrl -SessionVariable changeDataSession -UseBasicParsing | Out-Null
        }
        $downloadUrl = 'https://fbk.sharepoint.com/sites/BENCHMARKS/_layouts/15/download.aspx?UniqueId=' + $downloadIds[$epoch]
        Invoke-WebRequest -Uri $downloadUrl -WebSession $changeDataSession -OutFile $targetFile -UseBasicParsing
    }
    if ((Get-Item -LiteralPath $targetFile).Length -ne $expectedSizes[$epoch]) { throw 'Unexpected download size' }
    $fileHash = Get-FileHash -LiteralPath $targetFile -Algorithm SHA256
    if ($fileHash.Hash.ToLowerInvariant() -ne $expectedHashes[$epoch]) { throw 'Dataset hash changed; verify the official source before using it' }
    $fileHash
}

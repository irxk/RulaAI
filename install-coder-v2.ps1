$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$tabbyHome = if ($env:TABBYAPI_HOME) {
    $env:TABBYAPI_HOME
} else {
    Join-Path $env:LOCALAPPDATA "Programs\TabbyAPI"
}
$python = Join-Path $tabbyHome "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "TabbyAPI の Python が見つかりません: $python"
}

$infoJson = & $python -c "import json, sys, torch; print(json.dumps({'python': f'{sys.version_info.major}.{sys.version_info.minor}', 'torch': torch.__version__.split('+')[0], 'cuda': torch.version.cuda.replace('.', '') if torch.version.cuda else None}))"
if ($LASTEXITCODE -ne 0) {
    throw "TabbyAPI の Python/PyTorch 情報を取得できませんでした。"
}
$info = $infoJson | ConvertFrom-Json
if (-not $info.cuda) {
    throw "CUDA版PyTorchが必要です。TabbyAPIのCUDA環境を確認してください。"
}

$targetVersion = "0.3.2+cu$($info.cuda).torch$($info.torch)"
$versionCheck = @'
import importlib.metadata as metadata
try:
    print(metadata.version("exllamav2"))
except metadata.PackageNotFoundError:
    print("")
'@
$installedVersion = & $python -c $versionCheck
if ($installedVersion -eq $targetVersion) {
    $smokeTest = @'
import os
from pathlib import Path
import torch
os.add_dll_directory(str(Path(torch.__file__).resolve().parent / "lib"))
import exllamav2_ext
from exllamav2 import ExLlamaV2
print(f"ExLlamaV2 CUDA extension: {exllamav2_ext.__file__}")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
'@
    & $python -c $smokeTest
    if ($LASTEXITCODE -ne 0) {
        throw "導入済みExLlamaV2のCUDA拡張を読み込めませんでした。"
    }
    Write-Host "対応するExLlamaV2 wheelは導入済みです。" -ForegroundColor Green
    return
}

$release = Invoke-RestMethod `
    -Uri "https://api.github.com/repos/turboderp-org/exllamav2/releases/tags/v0.3.2" `
    -Headers @{ "User-Agent" = "RulaCoder-ExLlamaV2-Installer" }
$pythonTag = "cp$($info.python.Replace('.', ''))"
$wheelName = "exllamav2-0.3.2+cu$($info.cuda).torch$($info.torch)-$pythonTag-$pythonTag-win_amd64.whl"
$wheelAsset = $release.assets | Where-Object name -eq $wheelName | Select-Object -First 1
if (-not $wheelAsset -or -not $wheelAsset.digest.StartsWith("sha256:")) {
    throw "公式ExLlamaV2 0.3.2に、Python $($info.python) / PyTorch $($info.torch) / CUDA $($info.cuda) 用のWindows wheelがありません。"
}

$wheel = Join-Path $env:TEMP $wheelName
$expectedHash = $wheelAsset.digest.Substring(7).ToLowerInvariant()
if ((Test-Path $wheel) -and
    (Get-FileHash $wheel -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    Remove-Item -LiteralPath $wheel
}
if (-not (Test-Path $wheel)) {
    Write-Host "公式ExLlamaV2 wheelを取得しています: $wheelName" -ForegroundColor Cyan
    & curl.exe -L --fail --retry 5 --retry-delay 2 --retry-all-errors `
        -sS -o $wheel $wheelAsset.browser_download_url
    if ($LASTEXITCODE -ne 0) {
        throw "ExLlamaV2 wheelのダウンロードに失敗しました。"
    }
}
if ((Get-FileHash $wheel -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    Remove-Item -LiteralPath $wheel
    throw "取得したExLlamaV2 wheelのSHA-256が公式リリースと一致しません。"
}

Write-Host "ExLlamaV2のビルド済みwheelをTabbyAPI環境へ導入しています。" -ForegroundColor Cyan
& $python -m pip install --no-deps $wheel
if ($LASTEXITCODE -ne 0) {
    throw "ExLlamaV2 wheelを導入できませんでした。TabbyAPIのPython環境を確認してください。"
}

$smokeTest = @'
import os
from pathlib import Path
import torch
os.add_dll_directory(str(Path(torch.__file__).resolve().parent / "lib"))
import exllamav2_ext
from exllamav2 import ExLlamaV2
print(f"ExLlamaV2 CUDA extension: {exllamav2_ext.__file__}")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
'@
& $python -c $smokeTest
if ($LASTEXITCODE -ne 0) {
    throw "ExLlamaV2のCUDA拡張を読み込めませんでした。インストールログを確認してください。"
}

Write-Host "完了しました。RulaCoderを起動すると、Coder専用ExLlamaV2サーバーが自動で起動します。" -ForegroundColor Green

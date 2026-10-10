# Rula の Discord Bot を起動します。
#
# API サーバーとは別プロセスです。生成は API 側で1件ずつ直列に処理されるので、
# Bot は API が動いている間だけ動きます(先に ..\run_api.ps1 を起動してください)。
#
# 設定は1つ上の RulaAI\.env を共有します(APIキーがそこにあるため)。
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env"

if (-not (Test-Path $envFile)) {
    Write-Host "$envFile がありません。.env.example をコピーしてください。" -ForegroundColor Yellow
    exit 1
}

foreach ($name in @("DISCORD_TOKEN", "RULA_API_KEY")) {
    # Either source counts. discord_bot.py reads the environment first and only
    # falls back to .env, so demanding the file here would refuse to start over a
    # token that is already set as a Windows environment variable.
    $set = [bool][Environment]::GetEnvironmentVariable($name) -or
           (Select-String -Path $envFile -Pattern "^$name=.+" -Quiet)
    if (-not $set) {
        Write-Host "$name が未設定です(環境変数にも $envFile にもありません)。" -ForegroundColor Yellow
        Write-Host "設定方法は README の「Discord Bot」を参照してください。" -ForegroundColor Gray
        exit 1
    }
}

# The API's port lives in the same .env; read it rather than assuming 8000, or a
# moved API looks like a stopped one.
$portMatch = Select-String -Path $envFile -Pattern "^RULA_API_PORT=(\d+)" | Select-Object -First 1
$port = if ($env:RULA_API_PORT) { $env:RULA_API_PORT }
        elseif ($portMatch) { $portMatch.Matches[0].Groups[1].Value }
        else { "8000" }

try {
    $health = Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 5
} catch {
    Write-Host "APIサーバーに接続できません (127.0.0.1:$port)。" -ForegroundColor Yellow
    Write-Host "別のウィンドウで ..\run_api.ps1 を先に起動してください。" -ForegroundColor Gray
    exit 1
}
if ($health.status -ne "ok") {
    Write-Host "APIサーバーがまだモデルを読み込んでいます。少し待ってから再実行してください。" -ForegroundColor Yellow
    exit 1
}

Write-Host "API: $($health.weights) ($($health.backend))" -ForegroundColor Gray
Write-Host "Discord に接続します (Ctrl+C で停止)" -ForegroundColor Cyan
python discord_bot.py

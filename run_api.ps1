# Rula API を起動し、Cloudflare Tunnel 経由で外部から使えるようにします。
#
# API は 127.0.0.1 にしか待ち受けません。外部からの到達は cloudflared が担当し、
# こちらから Cloudflare へ接続しに行くため、ルーターのポート開放は不要です。
#
# setup_tunnel.ps1 を実行済みなら固定URLを、未実行なら trycloudflare の
# 使い捨てURLを使います。
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".env") -or -not (Select-String -Path ".env" -Pattern "RULA_API_KEY_HASH" -Quiet)) {
    Write-Host "APIキーが未発行です。python make_api_key.py を実行してください。" -ForegroundColor Yellow
    exit 1
}

$port = if ($env:RULA_API_PORT) { $env:RULA_API_PORT } else { "8000" }
$cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
$tunnelConfig = Join-Path $PSScriptRoot "tunnel.yml"
# Absent until setup_tunnel.ps1 has run, so index carefully rather than assuming a match.
$hostMatch = Select-String -Path ".env" -Pattern "^RULA_HOSTNAME=(.+)$" | Select-Object -First 1
$hostname = if ($hostMatch) { $hostMatch.Matches[0].Groups[1].Value.Trim() } else { $null }

Write-Host "APIサーバーを起動します (127.0.0.1:$port)" -ForegroundColor Cyan
$api = Start-Process python -ArgumentList "api.py" -PassThru -NoNewWindow

try {
    # Wait for the model before opening the tunnel, so the first request from
    # outside does not arrive at a server that is not ready yet.
    Write-Host "モデルの読み込みを待っています…" -ForegroundColor Gray
    $ready = $false
    foreach ($i in 1..120) {
        Start-Sleep -Seconds 2
        try {
            if ((Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 3).status -eq "ok") {
                $ready = $true; break
            }
        } catch { }
        if ($api.HasExited) { throw "APIサーバーが終了しました。ログを確認してください。" }
    }
    if (-not $ready) { throw "モデルの読み込みが完了しませんでした。" }
    Write-Host "準備完了。" -ForegroundColor Green

    if (-not $cloudflared) {
        Write-Host "`ncloudflared が無いためローカルのみで動作します。" -ForegroundColor Yellow
        Write-Host "  http://127.0.0.1:$port" -ForegroundColor Cyan
        Wait-Process -Id $api.Id
        return
    }

    if ((Test-Path $tunnelConfig) -and $hostname) {
        Write-Host "`n固定URL: https://$hostname" -ForegroundColor Green
        Write-Host "(Ctrl+C で API とトンネルの両方を停止します)`n" -ForegroundColor Gray
        & cloudflared tunnel --config $tunnelConfig run
    } else {
        Write-Host "`n使い捨てURLで公開します(起動ごとに変わります)。" -ForegroundColor Yellow
        Write-Host "固定URLにするには: .\setup_tunnel.ps1 -Hostname api.example.com`n" -ForegroundColor Gray
        & cloudflared tunnel --url "http://127.0.0.1:$port"
    }
}
finally {
    if ($api -and -not $api.HasExited) {
        Write-Host "`nAPIサーバーを停止します…" -ForegroundColor Gray
        Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    }
}

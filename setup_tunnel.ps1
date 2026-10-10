# 固定URLで公開するための Cloudflare Tunnel を作成します。
#
# trycloudflare の使い捨てURLと違い、名前付きトンネルは自分のドメインに紐づくので
# URLが変わりません。Cloudflareに登録済みのドメインが必要です。
#
#   .\setup_tunnel.ps1 -Hostname api.example.com
#
# 一度実行すれば以後は run_api.ps1 だけで同じURLが使えます。
param(
    [string]$Hostname = "",
    [string]$TunnelName = "rula-api"
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Not given: the one set up before (.env), so re-running needs no arguments.
if (-not $Hostname -and (Test-Path ".env")) {
    $m = Select-String -Path ".env" -Pattern '^\s*RULA_HOSTNAME\s*=\s*(\S+)' | Select-Object -First 1
    if ($m) { $Hostname = $m.Matches[0].Groups[1].Value }
}
if (-not $Hostname) {
    Write-Host "公開に使うホスト名を指定してください:" -ForegroundColor Yellow
    Write-Host "  .\setup_tunnel.ps1 -Hostname api.example.com" -ForegroundColor Yellow
    exit 1
}

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared がありません。次で導入してください:" -ForegroundColor Yellow
    Write-Host "  winget install --id Cloudflare.cloudflared" -ForegroundColor Yellow
    exit 1
}

$cfDir = Join-Path $env:USERPROFILE ".cloudflared"
$cert = Join-Path $cfDir "cert.pem"

# 1. Cloudflare アカウントに紐づける(ブラウザが開きます)
if (-not (Test-Path $cert)) {
    Write-Host "`n[1/4] Cloudflare にログインします。" -ForegroundColor Cyan
    Write-Host "ブラウザが開くので、$Hostname のドメインを選んで承認してください。`n" -ForegroundColor Gray
    cloudflared tunnel login
    if (-not (Test-Path $cert)) { throw "ログインが完了しませんでした。" }
} else {
    Write-Host "[1/4] ログイン済みです。" -ForegroundColor Green
}

# 2. トンネル本体を作る(認証情報が .cloudflared に保存される)
$existing = (cloudflared tunnel list 2>$null | Select-String -Pattern "\s$TunnelName\s")
if (-not $existing) {
    Write-Host "[2/4] トンネル '$TunnelName' を作成します。" -ForegroundColor Cyan
    cloudflared tunnel create $TunnelName
} else {
    Write-Host "[2/4] トンネル '$TunnelName' は作成済みです。" -ForegroundColor Green
}

$uuid = (cloudflared tunnel list --output json | ConvertFrom-Json |
         Where-Object { $_.name -eq $TunnelName } | Select-Object -First 1).id
if (-not $uuid) { throw "トンネルIDを取得できませんでした。" }

# 3. ホスト名をこのトンネルへ向ける(CloudflareにCNAMEが作られる)
Write-Host "[3/4] $Hostname をトンネルに接続します。" -ForegroundColor Cyan
cloudflared tunnel route dns --overwrite-dns $TunnelName $Hostname

# 4. 設定ファイル。ingress の最後は必ず catch-all が要る。
$port = if ($env:RULA_API_PORT) { $env:RULA_API_PORT } else { "8000" }
$credentials = Join-Path $cfDir "$uuid.json"
$config = @"
tunnel: $uuid
credentials-file: $credentials

ingress:
  - hostname: $Hostname
    service: http://127.0.0.1:$port
  - service: http_status:404
"@
$configPath = Join-Path $PSScriptRoot "tunnel.yml"
Set-Content -Path $configPath -Value $config -Encoding UTF8
Write-Host "[4/4] 設定を書き出しました: $configPath" -ForegroundColor Green

# ホスト名を .env に控えて、run_api.ps1 が拾えるようにする
$envPath = Join-Path $PSScriptRoot ".env"
$lines = if (Test-Path $envPath) { Get-Content $envPath | Where-Object { $_ -notmatch "^RULA_HOSTNAME=" } } else { @() }
Set-Content -Path $envPath -Value (($lines + "RULA_HOSTNAME=$Hostname") -join "`n") -Encoding UTF8

Write-Host "`n完了しました。固定URL: https://$Hostname" -ForegroundColor Green
Write-Host "起動するには: .\run_api.ps1" -ForegroundColor Cyan
Write-Host "DNSの反映に数分かかることがあります。" -ForegroundColor Gray

# PC起動時に Rula API を自動で立ち上げます(常時稼働用)。
#
# サービスではなくログオン時のタスクとして登録します。GPUを使うため、
# サービスのセッション0では動かないことがあるためです。
#
#   .\install_autostart.ps1            # 登録
#   .\install_autostart.ps1 -Remove    # 解除
param([switch]$Remove)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$taskName = "RulaAPI"

if ($Remove) {
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "自動起動を解除しました。" -ForegroundColor Green
    } else {
        Write-Host "登録されていません。" -ForegroundColor Yellow
    }
    return
}

if (-not (Test-Path ".env") -or -not (Select-String -Path ".env" -Pattern "RULA_API_KEY_HASH" -Quiet)) {
    Write-Host "先に python make_api_key.py を実行してください。" -ForegroundColor Yellow
    exit 1
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$PSScriptRoot\run_api.ps1`"" `
    -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
# Restart on failure, and never stop it for running too long or on battery.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Rula API + Cloudflare Tunnel" -Force | Out-Null

Write-Host "自動起動を登録しました: $taskName" -ForegroundColor Green
Write-Host "  今すぐ開始: Start-ScheduledTask -TaskName $taskName" -ForegroundColor Cyan
Write-Host "  状態確認  : Get-ScheduledTask -TaskName $taskName" -ForegroundColor Cyan
Write-Host "  解除      : .\install_autostart.ps1 -Remove" -ForegroundColor Cyan

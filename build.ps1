# Rula のアプリをビルドします(1秒前後)。
#
#   .\build.ps1             Rula.exe と Rula.lnk を作る
#   .\build.ps1 -Shortcut   さらにデスクトップとスタートメニューに「Rula」を置く
#   .\build.ps1 -NoExe      Rula.exe を作らない(Rula.lnk だけ)
#
# どちらをダブルクリックしても同じアプリが開きます。
#   Rula.exe — Windows 付属の C# コンパイラで作るランチャー。追加のインストール不要。
#              ただし署名の無い作りたての exe なので、ウイルス対策ソフトが誤検知して
#              隔離することがあります(この PC では Malwarebytes / Tencent PC Manager
#              が実行した瞬間に消しました)。
#   Rula.lnk — Python 本体(署名付きの pythonw.exe)を Rula のアイコンで起動する
#              ショートカット。exe を使わないので誤検知されません。
param([switch]$Shortcut, [switch]$NoExe)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$sw = [Diagnostics.Stopwatch]::StartNew()

# install.ps1 が用意する Rula 専用の Python があればそれを、無ければ PC の Python を使う。
$embeddedDir = Join-Path $PSScriptRoot "runtime\python"
$py = if (Test-Path "$embeddedDir\python.exe") { "$embeddedDir\python.exe" } else { "python" }

if (-not (Test-Path "launcher\rula.ico")) { & $py scripts\make_icon.py | Out-Null }
$icon = Join-Path $PSScriptRoot "launcher\rula.ico"

if (-not $NoExe) {
    $csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
    if (-not (Test-Path $csc)) { throw "C# コンパイラが見つかりません: $csc" }
    & $csc /nologo /target:winexe /optimize+ /out:Rula.exe /win32icon:launcher\rula.ico `
        /r:System.Windows.Forms.dll launcher\RulaLauncher.cs
    if ($LASTEXITCODE -ne 0) { throw "Rula.exe のビルドに失敗しました" }
}

$pythonw = if (Test-Path "$embeddedDir\pythonw.exe") { "$embeddedDir\pythonw.exe" }
           else { (Get-Command pythonw -ErrorAction SilentlyContinue).Source }
if (-not $pythonw) { throw "pythonw.exe が見つかりません (install.ps1 を実行するか、Python 3 をインストールしてください)" }
function New-RulaShortcut([string]$path) {
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
    $s.TargetPath = $pythonw
    $s.Arguments = "-m src.rula.gui"
    $s.WorkingDirectory = $PSScriptRoot
    $s.IconLocation = "$icon,0"
    $s.Description = "Rula — 日本語AIアシスタント"
    $s.Save()
}
New-RulaShortcut (Join-Path $PSScriptRoot "Rula.lnk")

$made = if ($NoExe) { "Rula.lnk" } else { "Rula.exe と Rula.lnk" }
Write-Host ("{0} を作りました ({1:N1} 秒)" -f $made, $sw.Elapsed.TotalSeconds) -ForegroundColor Green

if ($Shortcut) {
    foreach ($dir in [Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs")) {
        New-RulaShortcut (Join-Path $dir "Rula.lnk")
    }
    Write-Host "デスクトップとスタートメニューに「Rula」を置きました" -ForegroundColor Green
}
Write-Host $(if ($NoExe) { "起動: Rula.lnk をダブルクリック" } else { "起動: Rula.lnk (または Rula.exe) をダブルクリック" })

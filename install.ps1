# Rula を導入します。Rula-Setup.cmd をダブルクリックすると、これが動きます。
#
#   必要なもの: Windows 10/11 (64bit)、インターネット接続
#   Main RulaにはTabbyAPIとLlamaモデル、RulaCoderにはEXL2モデルとExLlamaV2 wheelが必要です。
#   このスクリプトは推論サーバーやモデルをダウンロードしません。
#
# PC の設定は変えません。すべてこのフォルダの中に入ります(会話の履歴だけ
# %APPDATA%\Rula)。消すときは、このフォルダと %APPDATA%\Rula を削除してください。
# 途中で止まっても、もう一度実行すれば続きから再開します。
#
#   .\install.ps1 -NoShortcut     デスクトップとスタートメニューにショートカットを置かない
#   .\install.ps1 -DesktopExe     デスクトップに Rula.exe を置く (1行インストールの既定)
param([switch]$NoShortcut, [switch]$DesktopExe, [switch]$NoLaunch)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$ProgressPreference = "SilentlyContinue"  # Expand-Archive の進捗表示は遅い

# Rula 専用の Python。python.org 公式の埋め込み版と PyPI の pip を、SHA256 を照合して使う。
$PY_VERSION = "3.13.7"
$PY_ZIP = "python-$PY_VERSION-embed-amd64.zip"
$PY_URL = "https://www.python.org/ftp/python/$PY_VERSION/$PY_ZIP"
$PY_SHA = "f6cca216a359be84797cabb54149ce5e062afb16cc7567eb7fc51cacb2d86b65"
$PIP_WHL = "pip-26.0.1-py3-none-any.whl"
$PIP_URL = "https://files.pythonhosted.org/packages/de/f0/c81e05b613866b76d2d1066490adf1a3dbc4ee9d9c839961c3fc8a6997af/$PIP_WHL"
$PIP_SHA = "bdb1b08f4274833d62c1aa29e20907365a2ceb950410df15fc9521bad440122b"

function Say($text, $color = "White") { Write-Host $text -ForegroundColor $color }
function Fail($text) { throw $text }

function Get-File($url, $out, $sha) {
    $name = Split-Path $out -Leaf
    if ((Test-Path $out) -and (Get-FileHash $out -Algorithm SHA256).Hash.ToLower() -eq $sha) { return }
    Say "  取得中: $name" Cyan
    & curl.exe -L --fail --retry 10 --retry-delay 3 --retry-all-errors -sS -C - -o $out $url
    if ($LASTEXITCODE -ne 0) { Fail "ダウンロードに失敗しました: $name (回線を確かめて、もう一度実行してください)" }
    if ((Get-FileHash $out -Algorithm SHA256).Hash.ToLower() -ne $sha) {
        Remove-Item $out
        Fail "取得したファイルが配布元のものと一致しません: $name (もう一度実行してください)"
    }
}

Say ""
Say "=== Rula のセットアップ ===" Cyan
Say ""
Say "Rula は TabbyAPI (ExLlamaV3) に接続して動く日本語 AI アシスタントです。"
Say "標準設定の接続先はこの PC の TabbyAPI です。別の接続先では会話内容がそのサーバーに送られます。"
Say "(質問に応じて Web 検索をするときは、検索語が検索エンジンに送られます)。"
try {
if (-not [Environment]::Is64BitOperatingSystem) { Fail "64bit 版の Windows が必要です。" }

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $dl = Join-Path $PSScriptRoot "runtime\downloads"
    $pyDir = Join-Path $PSScriptRoot "runtime\python"
    New-Item -ItemType Directory -Force -Path $dl | Out-Null

    # -- 1. Rula 専用の Python ------------------------------------------------
    Say "[1/2] Python を用意しています" Cyan
    $py = Join-Path $pyDir "python.exe"
    if (-not (Test-Path $py)) {
        Get-File $PY_URL "$dl\$PY_ZIP" $PY_SHA
        Expand-Archive -Path "$dl\$PY_ZIP" -DestinationPath $pyDir -Force
        # 埋め込み版は既定で site-packages も作業フォルダも読まない。Rula 本体
        # (このフォルダ) と、pip で入れる部品の場所を読むようにする。
        $pth = Get-ChildItem "$pyDir\python3*._pth" | Select-Object -First 1
        $zipName = (Get-ChildItem "$pyDir\python3*.zip" | Select-Object -First 1).Name
        Set-Content -Path $pth.FullName -Encoding ascii -Value @($zipName, ".", "..\..", "Lib\site-packages", "import site")
        # 利用者が別に入れている Python の部品 (AppData\Roaming\Python) は読まない。
        # 読むと、同じ部品の別の版が混ざって動かなくなることがある(試験で実際に混ざった)。
        Set-Content -Path "$pyDir\sitecustomize.py" -Encoding ascii -Value @(
            "# Rula's own Python uses only its own packages, never the user's site-packages.",
            "import site, sys",
            "_user = site.getusersitepackages()",
            "sys.path[:] = [p for p in sys.path if not p.startswith(_user)]",
            "site.ENABLE_USER_SITE = False")
    }
    $lock = Join-Path $PSScriptRoot "requirements.lock"
    $marker = Join-Path $pyDir ".rula-deps"
    $want = (Get-FileHash $lock -Algorithm SHA256).Hash
    if (-not ((Test-Path $marker) -and (Get-Content $marker -Raw).Trim() -eq $want)) {
        # (Checked on disk: in Windows PowerShell, a native command's stderr
        # redirected under ErrorActionPreference=Stop is a terminating error.)
        if (-not (Test-Path "$pyDir\Lib\site-packages\pip")) {
            Get-File $PIP_URL "$dl\$PIP_WHL" $PIP_SHA
            # wheel は zip そのもの。pip に pip 自身を入れさせると Windows では
            # 「自分を書き換えようとしている」と止まるので、展開して置く。
            $zip = "$dl\pip-wheel.zip"
            Copy-Item "$dl\$PIP_WHL" $zip -Force
            Expand-Archive -Path $zip -DestinationPath "$pyDir\Lib\site-packages" -Force
            Remove-Item $zip
        }
        # 版を固定し、組み立て済みの部品 (wheel) だけを使う: コンパイラの無い PC でも入る。
        & $py -m pip install --quiet --disable-pip-version-check --no-warn-script-location `
            --only-binary=:all: -r $lock
        if ($LASTEXITCODE -ne 0) { Fail "部品のインストールに失敗しました (回線を確かめて、もう一度実行してください)" }
        Set-Content -Path $marker -Value $want -Encoding ascii
    }
    Say "  OK" Green

    # -- 2. 起動用のショートカット --------------------------------------------
    Say "[2/2] 起動用のファイルを作っています" Cyan
    $build = Join-Path $PSScriptRoot "build.ps1"
    if ($NoShortcut) { & $build }
    elseif ($DesktopExe) { & $build -DesktopExe }
    else { & $build -Shortcut }
    $desktop = [Environment]::GetFolderPath("Desktop")
    $launcher = if ($DesktopExe -and -not $NoShortcut) { Join-Path $desktop "Rula.exe" }
                elseif ($NoShortcut) { Join-Path $PSScriptRoot "Rula.exe" }
                else { Join-Path $PSScriptRoot "Rula.lnk" }

    Say ""
    Say ("=== 完了しました ({0:N0} 分) ===" -f $sw.Elapsed.TotalMinutes) Green
    Say "Rulaフォルダの start-tabbyapi.bat でMain用TabbyAPIを起動してください。" Yellow
    Say "RulaCoder用には install-coder-v2.ps1 を一度実行してください。" Yellow
    if ($DesktopExe -and -not $NoShortcut) {
        Say "次からは、デスクトップの Rula.exe (またはスタートメニューの「Rula」) で起動します。"
    } else {
        Say "次からは、デスクトップかスタートメニューの「Rula」、またはこのフォルダの Rula.lnk で起動します。"
    }
    if (-not $NoLaunch) {
        Say "Rula を起動します (先にstart-tabbyapi.batでTabbyAPIを起動してください)。"
        $since = Get-Date
        $started = $true
        try { $proc = Start-Process $launcher -PassThru } catch { $started = $false }
        # 署名の無い作りたての exe は、ウイルス対策ソフトに消されたり止められたりする
        # ことがある (開発者の PC で実際に起きた: 起動した瞬間に消え、アプリは開かな
        # かった)。Rula が立ち上がった印 (running.txt) が出るまで見届け、出ないうちに
        # exe が消えるか止まったら、誤検知されないショートカットに替える。
        if ($launcher.EndsWith(".exe")) {
            $runFile = Join-Path $env:APPDATA "Rula\running.txt"
            $ok = $false
            for ($i = 0; $started -and $i -lt 80; $i++) {
                if ((Test-Path $runFile) -and (Get-Item $runFile).LastWriteTime -ge $since) { $ok = $true; break }
                if (-not (Test-Path $launcher)) { break }
                if ($proc.HasExited) {
                    # Rula was already open: this exe opened another window onto it and ended.
                    if ($proc.ExitCode -eq 0 -and (Test-Path $runFile)) { $ok = $true }
                    break
                }
                Start-Sleep -Milliseconds 500
            }
            # Still starting after 40 s but the exe is there and running: leave it be.
            if (-not $ok -and $started -and (Test-Path $launcher) -and -not $proc.HasExited) { $ok = $true }
            if (-not $ok) {
                if ($started -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
                Remove-Item $launcher -ErrorAction SilentlyContinue
                Remove-Item (Join-Path $PSScriptRoot "RulaCoder.exe") -ErrorAction SilentlyContinue
                Remove-Item (Join-Path $desktop "RulaCoder.exe") -ErrorAction SilentlyContinue
                & $build -NoExe -Shortcut | Out-Null
                Say ""
                Say "デスクトップの Rula.exe がウイルス対策ソフトに止められました (署名の無い exe への誤検知です)。" Yellow
                Say "代わりにデスクトップへショートカット「Rula」を置きました。こちらから起動してください。" Yellow
                Start-Process (Join-Path $PSScriptRoot "Rula.lnk")
            }
        }
    }
} catch {
    Say ""
    Say "セットアップが止まりました: $($_.Exception.Message)" Red
    Say "もう一度 Rula-Setup.cmd を実行すると、取得済みのファイルはそのまま、続きから再開します。" Yellow
    exit 1
}

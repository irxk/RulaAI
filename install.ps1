# Rula を導入します。Rula-Setup.cmd をダブルクリックすると、これが動きます。
#
#   必要なもの: Windows 10/11 (64bit)、NVIDIA の GPU (VRAM 8GB 以上で 9B、6GB 以上で 4B)、
#               空きディスク 約14GB (導入後は約8GB)、インターネット接続
#   所要時間:   回線によって 10〜40 分 (ほとんどがモデルのダウンロード)
#
# PC の設定は変えません。すべてこのフォルダの中に入ります(会話の履歴だけ
# %APPDATA%\Rula)。消すときは、このフォルダと %APPDATA%\Rula を削除してください。
# 途中で止まっても、もう一度実行すれば続きから再開します。
#
#   .\install.ps1                 VRAM に合わせてモデルを選ぶ
#   .\install.ps1 -Model 4b       4B に固定
#   .\install.ps1 -NoShortcut     デスクトップとスタートメニューにショートカットを置かない
#   .\install.ps1 -DesktopExe     デスクトップに Rula.exe を置く (1行インストールの既定)
param([ValidateSet("auto", "9b", "4b")][string]$Model = "auto",
      [switch]$NoShortcut, [switch]$DesktopExe, [switch]$NoLaunch, [switch]$Yes)
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
Say "Rula は、この PC の GPU だけで動く日本語 AI アシスタントです。会話の内容は外に送りません"
Say "(質問に応じて Web 検索をするときは、検索語が検索エンジンに送られます)。"
Say ""
Say "ご注意:" Yellow
Say "  会話モデルは、安全のための調整を緩めた「無検閲 (uncensored)」版です。" Yellow
Say "  拒否のルールはプロンプトで与えているだけで、モデルの重みには含まれていません。" Yellow
Say "  詳しくは NOTICE をお読みください。" Yellow
Say ""

try {
# -- この PC で動くかを先に確かめる ------------------------------------------
if (-not [Environment]::Is64BitOperatingSystem) { Fail "64bit 版の Windows が必要です。" }
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Fail "NVIDIA の GPU とドライバーが見つかりません。Rula は NVIDIA の GPU (VRAM 6GB 以上) で動きます。"
}
$gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits |
        Select-Object -First 1) -split ",\s*"
$gpuName, $vram, $driver = $gpu[0], [int]$gpu[1], $gpu[2]
Say ("GPU: {0} (VRAM {1:N1}GB, ドライバー {2})" -f $gpuName, ($vram / 1024), $driver)
# llama.cpp の CUDA 12.4 版は、ドライバー 551.61 以降で動く。
if ([version]$driver -lt [version]"551.61") {
    Fail "NVIDIA のドライバーが古いため動きません ($driver)。551.61 以降に更新してください (NVIDIA App または nvidia.com から)。"
}
if ($Model -eq "auto") {
    if ($vram -ge 7800) { $Model = "9b" }
    elseif ($vram -ge 5800) { $Model = "4b" }
    else { Fail ("VRAM が足りません ({0:N1}GB)。6GB 以上の NVIDIA GPU が必要です。" -f ($vram / 1024)) }
}
# 作業中に最も膨らむ量 (元のモデルと MTP 付きが並ぶ時)。取得済みのぶんは差し引く:
# 途中から再開するときに、もう置いてあるファイルの分まで空きを求めない。
$needGB = if ($Model -eq "9b") { 14 } else { 8 }
$haveBytes = (Get-ChildItem (Join-Path $PSScriptRoot "models"), (Join-Path $PSScriptRoot "runtime") `
                -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
$needGB = [math]::Max(1, [math]::Ceiling($needGB - [double]$haveBytes / 1GB))
$drive = (Get-Item $PSScriptRoot).PSDrive
$freeGB = [math]::Floor($drive.Free / 1GB)
Say "モデル: $Model    空きディスク: ${freeGB}GB (必要: 約${needGB}GB)"
if ($freeGB -lt $needGB) { Fail "ディスクの空きが足りません。$($drive.Name): ドライブを ${needGB}GB 以上空けてから、もう一度実行してください。" }
Say ""
if (-not $Yes) {
    Say "続けるとダウンロードを始めます (合計 約$(if ($Model -eq '9b') {'7'} else {'4'})GB)。" Cyan
    $null = Read-Host "Enter で開始 (やめるときはこのウィンドウを閉じてください)"
}

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $dl = Join-Path $PSScriptRoot "runtime\downloads"
    $pyDir = Join-Path $PSScriptRoot "runtime\python"
    New-Item -ItemType Directory -Force -Path $dl | Out-Null

    # -- 1. Rula 専用の Python ------------------------------------------------
    Say "[1/3] Python を用意しています" Cyan
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

    # -- 2. 推論エンジンとモデル ----------------------------------------------
    Say "[2/3] 推論エンジンとモデルを取得しています (時間がかかります)" Cyan
    & (Join-Path $PSScriptRoot "setup_llama.ps1") -Model $Model

    # -- 3. 起動用のショートカット --------------------------------------------
    Say "[3/3] 起動用のファイルを作っています" Cyan
    $build = Join-Path $PSScriptRoot "build.ps1"
    if ($NoShortcut) { & $build -NoExe }
    elseif ($DesktopExe) { & $build -DesktopExe }
    else { & $build -NoExe -Shortcut }
    $desktop = [Environment]::GetFolderPath("Desktop")
    $launcher = if ($DesktopExe -and -not $NoShortcut) { Join-Path $desktop "Rula.exe" }
                else { Join-Path $PSScriptRoot "Rula.lnk" }

    Say ""
    Say ("=== 完了しました ({0:N0} 分) ===" -f $sw.Elapsed.TotalMinutes) Green
    if ($DesktopExe -and -not $NoShortcut) {
        Say "次からは、デスクトップの Rula.exe (またはスタートメニューの「Rula」) で起動します。"
    } else {
        Say "次からは、デスクトップかスタートメニューの「Rula」、またはこのフォルダの Rula.lnk で起動します。"
    }
    if (-not $NoLaunch) {
        Say "Rula を起動します (初回はモデルの読み込みに 10 秒ほどかかります)。"
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

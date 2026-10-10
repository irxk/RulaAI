# llama-server で動かすための一式を取得します(初回のみ)。
#
#   llama.cpp 本体 (b11172, CUDA 12.4 版)   243MB + CUDA ランタイム 373MB
#   会話モデル Q4_K_M                       9B: 5.4GB / 4B: 2.6GB
#   画像を読むための部品 (mmproj)           9B: 879MB / 4B: 644MB   -NoVision で省略
#   MTP 層 (生成速度 +45% 前後)             9B: 160MB / 4B: 77MB    -NoMtp で省略
#
# 既定は 9B(品質重視)。4B は速度と VRAM 重視: .\setup_llama.ps1 -Model 4b
# 取得したファイルはすべて配布元が公開している SHA256 と照合します。
# MTP 層の移植には pip の gguf パッケージ (ggml-org 公式) を使います。
# MTP 付きのモデルを作って照合できたら、元のモデルは消します(-KeepBase で残す)。
# MTP 付きのファイルは MTP を使わない起動にもそのまま使えるため(実測)、5.4GB 減ります。
param([ValidateSet("9b", "4b")][string]$Model = "9b", [switch]$NoMtp, [switch]$NoVision,
      [switch]$KeepBase)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# install.ps1 が用意する Rula 専用の Python があればそれを、無ければ PC の Python を使う。
$embedded = Join-Path $PSScriptRoot "runtime\python\python.exe"
$py = if (Test-Path $embedded) { $embedded } else { "python" }

$runtime = Join-Path $PSScriptRoot "runtime\llama.cpp"
$gguf = Join-Path $PSScriptRoot "models\gguf"
$zips = Join-Path $PSScriptRoot "runtime\zips"
New-Item -ItemType Directory -Force -Path $runtime, $gguf, $zips | Out-Null

$spec = @{
    "9b" = @{ repo = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"; stem = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
              sha = "2ca636d9e81d3d23ca9b60c234fe185d30ec082eeba69ce770fdb0c76559a4f5"
              mmsha = "05f662501f8bd45607b079723a3e238a4e888fd085a10a53f4057a0e250f6934"
              mtp = "https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf"
              mtpsha = "7eb40e02590190dfc7ed827675760f34514d9fb25a84aef3515c4412cad29403" }
    "4b" = @{ repo = "HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive"; stem = "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive"
              sha = "79e28ecacf84e75b6056cf4059636d435aa9eb67795780f7b7dbc7d32a962741"
              mmsha = "a1e32e86ea99aa7a56f3dcfe7e63c1d0be9439d31fd07087099f15bc0fda0f22"
              mtp = "https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
              mtpsha = "b7db096f5efa8db29f9f33aa8aa2d418033552293ac17701c6f86a4b407f9d01" }
}[$Model]
$hf = "https://huggingface.co/$($spec.repo)/resolve/main"
$rel = "https://github.com/ggml-org/llama.cpp/releases/download/b11172"
$base = "$gguf\rula-$Model-q4_k_m.gguf"
$mtpOut = "$gguf\rula-$Model-mtp-q4_k_m.gguf"

function Get-Sha($path) { (Get-FileHash $path -Algorithm SHA256).Hash.ToLower() }

# 照合が済んだ印を、ファイルの隣に .ok として残す(期待する SHA・サイズ・更新日時)。
function Write-Ok($path, $sha) {
    $f = Get-Item $path
    Set-Content -Path "$path.ok" -Encoding ascii -Value "$sha|$($f.Length)|$($f.LastWriteTimeUtc.Ticks)"
}

# ファイルがあって $sha と一致するか。.ok 印があり、サイズと更新日時が印の時と
# 同じなら、ハッシュを取り直さずに信用する(数GBの再ハッシュを毎回避ける)。
# SHA256 自体は、ダウンロード時に1回必ず検証し、印が無い/ファイルが変わった時は
# ここでも取り直す。スクリプトの期待 SHA が変わった(モデルを更新した)時も、
# 印の SHA と食い違うので取り直しになる。
function Test-Verified($path, $sha) {
    if (-not (Test-Path $path)) { return $false }
    $f = Get-Item $path
    $mark = "$path.ok"
    if (Test-Path $mark) {
        $p = (Get-Content $mark -TotalCount 1) -split '\|'
        if ($p.Count -eq 3 -and $p[0] -eq $sha -and $p[1] -eq "$($f.Length)" `
                -and $p[2] -eq "$($f.LastWriteTimeUtc.Ticks)") {
            return $true
        }
    }
    if ((Get-Sha $path) -eq $sha) { Write-Ok $path $sha; return $true }
    return $false
}

# MTP 付きが照合済みなら元のモデルは要らない(移植の材料としてだけ使う)。
$mtpReady = (-not $NoMtp) -and (Test-Verified $mtpOut $spec.mtpsha)

$items = @()
# 展開済みなら zip は要らない(展開後に消している)。
if (-not (Test-Path (Join-Path $runtime "llama-server.exe"))) {
    $items += @{ url = "$rel/llama-b11172-bin-win-cuda-12.4-x64.zip"; out = "$zips\llama-b11172-bin-win-cuda-12.4-x64.zip"
                 sha = "bdbbec11381cf896188cdcc8a46e7fb895fa3e135fc8609d32ce3e8a5a42f181" }
    $items += @{ url = "$rel/cudart-llama-bin-win-cuda-12.4-x64.zip"; out = "$zips\cudart-llama-bin-win-cuda-12.4-x64.zip"
                 sha = "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6" }
}
if (-not $mtpReady) {
    $items += @{ url = "$hf/$($spec.stem)-Q4_K_M.gguf"; out = $base; sha = $spec.sha }
}
if (-not $NoVision) {
    $items += @{ url = "$hf/mmproj-$($spec.stem)-BF16.gguf"; out = "$gguf\mmproj-rula-$Model-bf16.gguf"; sha = $spec.mmsha }
}

function Get-Verified($it) {
    $name = Split-Path $it.out -Leaf
    if (Test-Verified $it.out $it.sha) {
        Write-Host "  取得済み: $name" -ForegroundColor Gray
        return
    }
    Write-Host "  取得中: $name" -ForegroundColor Cyan
    # 再開(-C -)と停止検出: 50KB/s 未満が30秒続いたら切り直して続きから取る。
    & curl.exe -L --fail --retry 20 --retry-delay 3 --retry-all-errors `
        --speed-limit 51200 --speed-time 30 -C - -o $it.out $it.url
    if ($LASTEXITCODE -ne 0) { throw "ダウンロードに失敗しました: $name" }
    if ((Get-Sha $it.out) -ne $it.sha) { throw "SHA256 が一致しません: $name (ファイルを消してやり直してください)" }
    Write-Ok $it.out $it.sha
    Write-Host "  照合OK: $name" -ForegroundColor Green
}

Write-Host "ダウンロード ($Model)" -ForegroundColor Cyan
foreach ($it in $items) { Get-Verified $it }

if (-not (Test-Path (Join-Path $runtime "llama-server.exe"))) {
    Write-Host "llama.cpp を展開しています" -ForegroundColor Cyan
    # 本体とランタイムを同じ場所へ: CUDA の DLL は llama-server.exe の隣に要る。
    foreach ($z in Get-ChildItem "$zips\*.zip") { Expand-Archive -Path $z.FullName -DestinationPath $runtime -Force }
}

if (-not $NoMtp) {
    # 移植は決定的なので、どのPCでも同じファイルになる。
    if ($mtpReady) {
        Write-Host "MTP 付きモデルは作成済みです" -ForegroundColor Gray
    } else {
        Write-Host "MTP 層を移植しています" -ForegroundColor Cyan
        & $py -m pip install --quiet --disable-pip-version-check "gguf==0.19.0"
        & $py scripts\graft_mtp.py $base $spec.mtp $mtpOut (Join-Path $gguf ".mtp-work")
        if ($LASTEXITCODE -ne 0) { throw "MTP 層の移植に失敗しました" }
        if ((Get-Sha $mtpOut) -ne $spec.mtpsha) {
            Remove-Item $mtpOut
            throw "MTP 付きモデルの SHA256 が想定と違います。移植版は使わずに続行できます (-NoMtp)。"
        }
        Write-Ok $mtpOut $spec.mtpsha
        Write-Host "  照合OK: rula-$Model-mtp-q4_k_m.gguf" -ForegroundColor Green
        # 移植の作業用 (MTP 層の取得分)。照合が済めば要らない。
        $work = Join-Path $gguf ".mtp-work"
        if (Test-Path $work) { Remove-Item -LiteralPath $work -Recurse -Force }
    }
    if ((Test-Path $base) -and -not $KeepBase) {
        Remove-Item $base
        Remove-Item "$base.ok" -ErrorAction SilentlyContinue
        Write-Host "  元のモデルを削除しました (MTP 付きだけで動きます。-KeepBase で残せます)" -ForegroundColor Gray
    }
}
# 取得用の zip は展開済みなので要らない。
if (Test-Path (Join-Path $runtime "llama-server.exe")) {
    Get-ChildItem "$zips\*.zip" -ErrorAction SilentlyContinue | Remove-Item
}

Write-Host "`n完了。" -ForegroundColor Green
if ($Model -eq "4b") {
    $final = if ($NoMtp) { $base } else { "$gguf\rula-4b-mtp-q4_k_m.gguf" }
    Write-Host "9B のファイルがあればそちらが優先されます。4B に固定するには .env に次を書いてください:" -ForegroundColor Yellow
    Write-Host "  RULA_GGUF=$final" -ForegroundColor Yellow
}

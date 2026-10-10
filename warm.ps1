# モデルを“温めて”常駐させます。
#
# このウィンドウを開いている間、Rula の起動が毎回一瞬になります
# (起動のたびに 9B モデルを読み込み直す約10秒を省けます)。
# そのあいだ VRAM を約6GB 使い続けます。終了して解放するには、
# このウィンドウで Ctrl+C を押すか、ウィンドウを閉じてください。
Set-Location -Path $PSScriptRoot
$embedded = Join-Path $PSScriptRoot "runtime\python\python.exe"
$py = if (Test-Path $embedded) { $embedded } else { "python" }
& $py -m src.rula.serve

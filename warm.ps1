# TabbyAPIが起動済みかを確認します。モデルの常駐管理はTabbyAPI側が行います。
Set-Location -Path $PSScriptRoot
$embedded = Join-Path $PSScriptRoot "runtime\python\python.exe"
$py = if (Test-Path $embedded) { $embedded } else { "python" }
& $py -m src.rula.serve

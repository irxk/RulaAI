# Rula one-line installer. In PowerShell:
#
#   iwr -useb https://raw.githubusercontent.com/irxk/RulaAI/main/setup.ps1 | iex
#
# Downloads Rula into %LOCALAPPDATA%\Programs\Rula (or $env:RULA_HOME if set),
# then runs install.ps1 there: it prepares Rula's own Python and puts Rula.exe and
# RulaCoder.exe on the desktop. Install Main's TabbyAPI, both model files, and the Coder wheel separately.
# Running this again updates Rula's code and keeps its Python environment.
#
# Plain ASCII, no BOM: run through Invoke-Expression, a BOM breaks the first
# line, and Windows PowerShell 5.1 misreads non-ASCII in a file without one.
# The Japanese messages come from install.ps1, which is run as a file.
& {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    # Older Windows PowerShell may not offer TLS 1.2 by default; GitHub needs it.
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $repo = 'irxk/RulaAI'
    $branch = 'main'
    $dir = if ($env:RULA_HOME) { $env:RULA_HOME } else { Join-Path $env:LOCALAPPDATA 'Programs\Rula' }

    Write-Host ''
    Write-Host "Rula: downloading the latest version into $dir" -ForegroundColor Cyan
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ('rula-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $zip = Join-Path $tmp 'rula.zip'
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/$repo/archive/refs/heads/$branch.zip" -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $tmp
        $src = Get-ChildItem -Path $tmp -Directory | Select-Object -First 1
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        # The code is replaced; runtime\ (Rula's Python environment) stays.
        Copy-Item -Path (Join-Path $src.FullName '*') -Destination $dir -Recurse -Force
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }

    # install.ps1 runs in its own Windows PowerShell, with the execution policy
    # bypassed for that one process. No system setting is changed.
    $ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    & $ps -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dir 'install.ps1') -DesktopExe
}

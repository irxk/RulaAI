@echo off
rem Rula setup. Double-click to run.
rem The execution policy is bypassed for install.ps1 in this window only; no system setting is changed.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause

@echo off
setlocal

set "TABBY_DIR=%LOCALAPPDATA%\Programs\TabbyAPI"
if defined TABBYAPI_HOME set "TABBY_DIR=%TABBYAPI_HOME%"
if not exist "%TABBY_DIR%\start.bat" (
    echo TabbyAPI start.bat was not found: "%TABBY_DIR%"
    exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%TABBY_DIR%" || exit /b 1
call start.bat %*
exit /b %errorlevel%

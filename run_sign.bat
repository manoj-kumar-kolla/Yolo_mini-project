@echo off
title AI Live Sign Language to Speech
echo ========================================================
echo   Launching AI Live Sign Language Generator...
echo ========================================================
echo.

:: Try Python 3.14 launcher first (where packages are installed)
py -3.14 "%~dp0sign.py"
if %ERRORLEVEL% EQU 0 goto end

echo.
echo Trying standard python launcher...
py "%~dp0sign.py"
if %ERRORLEVEL% EQU 0 goto end

echo.
echo Trying system PATH python...
python "%~dp0sign.py"

:end
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Program exited with an error.
    pause
)

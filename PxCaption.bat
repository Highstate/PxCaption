@echo off
rem PxCaption launch script (Windows).
rem - Uses the venv Python directly (robust to a failed "call activate").
rem - Runs unbuffered (-u) so a native traceback / log line appears immediately.
rem - "pause"s ONLY on a non-zero exit: a crash leaves the console open with the
rem   traceback; a normal GUI close lets the console window go away.
setlocal
set "PROJ=%~dp0"
set "VENV_PY=%PROJ%venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [PxCaption] Python venv not found at %VENV_PY%
    echo   Create it with:  python -m venv venv   then   pip install -r Requirements.txt
    pause
    exit /b 1
)
"%VENV_PY%" -u "%PROJ%PxCaption.py"
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo [PxCaption] exited with code %EXITCODE% - see the log pane / traceback above.
    pause
)
endlocal

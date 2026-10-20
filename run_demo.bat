@echo off
echo Starting Smart Solar Rail Track Fault Detection System...

:: Start Backend in a minimized window
if exist ".\venv\Scripts\python.exe" (
    start /min "Railway Backend" cmd /c ".\venv\Scripts\python.exe main.py"
) else (
    start /min "Railway Backend" cmd /c "python main.py"
)

echo Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

:: Open the Dashboard in the default browser
echo Opening Dashboard...
start http://localhost:8000

echo.
echo ======================================================
echo SYSTEM IS READY!
echo ======================================================
echo The web dashboard has been opened in your browser.
echo You can control the simulation directly from the UI.
echo.
echo Press any key to shutdown the server...
pause > nul

:: Kill the python process on exit
taskkill /F /IM python.exe /T > nul 2>&1

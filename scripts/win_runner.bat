@echo off
rem ------------------------------------------------------------------------
rem win_runner.bat -- job runner for the hackathon venv (Windows side).
rem
rem Double-click once and leave the window open.  It watches
rem   qhack\.runner\queue\*.cmd
rem and runs each job in turn with the hackathon venv
rem   ..\pasqal\Scripts\python.exe   (available to jobs as %PY%)
rem from the repo root (%ROOT%), writing
rem   .runner\out\<job>.log   (stdout+stderr)   and
rem   .runner\out\<job>.rc    (exit code, written when the job is done).
rem Close the window to stop it.  Nothing here touches the network except
rem what a job itself does (e.g. pip install).
rem ------------------------------------------------------------------------
setlocal enabledelayedexpansion
set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "PY=%ROOT%\..\pasqal\Scripts\python.exe"
for %%I in ("%PY%") do set "PY=%%~fI"
set "Q=%ROOT%\.runner\queue"
set "R=%ROOT%\.runner\running"
set "O=%ROOT%\.runner\out"
mkdir "%Q%" 2>nul
mkdir "%R%" 2>nul
mkdir "%O%" 2>nul
if not exist "%PY%" (
  echo ERROR: venv python not found at "%PY%"
  pause
  exit /b 1
)
echo runner up: ROOT=%ROOT%
echo             PY=%PY%
echo %date% %time% > "%O%\runner.alive"
cd /d "%ROOT%"

:loop
for /f "delims=" %%f in ('dir /b /o:n "%Q%\*.cmd" 2^>nul') do (
  move "%Q%\%%f" "%R%\%%f" >nul
  echo [%time%] START %%f
  cd /d "%ROOT%"
  call "%R%\%%f" > "%O%\%%~nf.log" 2>&1
  set "RC=!errorlevel!"
  echo !RC! > "%O%\%%~nf.rc"
  echo [%time%] DONE  %%f  rc=!RC!
  del "%R%\%%f" 2>nul
)
echo %date% %time% > "%O%\runner.alive"
timeout /t 3 /nobreak >nul
goto loop

@echo off
setlocal enabledelayedexpansion
set PROGRAMFILE=silero-tts-rt-server.py
set FOLDERNAME=silero-tts-rt-server

cd ..

set VERSION=
for /f "tokens=2 delims=^= " %%a in ('findstr "MAIN_VERSION" %PROGRAMFILE% 2^>nul') do (
    if not defined VERSION (
        set VERSION=%%a
        set VERSION=!VERSION:"=!
    )
)
if not defined VERSION set VERSION=0.4
set RELEASE_DIR=%FOLDERNAME%-%VERSION%

echo %FOLDERNAME% v%VERSION%

rmdir /s /q build dist __pycache__ 2>nul
del /s /q *.pyc *.spec *.manifest 2>nul
rmdir /s /q "%RELEASE_DIR%" 2>nul

c:\Python38-64\python.exe -m PyInstaller ^
    --onedir ^
    --noupx ^
    --icon=scripts\icon.ico ^
    --exclude-module tensorboard ^
    --exclude-module pytest ^
    %PROGRAMFILE%

xcopy "models\v5_5_ru.pt" "dist\%FOLDERNAME%\models\" /I
xcopy "README.md" "dist\%FOLDERNAME%\"
xcopy "tts-rt-server-simple-tester.html" "dist\%FOLDERNAME%\"
xcopy "LunaTranslator\*" "dist\%FOLDERNAME%\LunaTranslator\" /E /I

rename dist\%FOLDERNAME% "%RELEASE_DIR%"
if not exist releases mkdir releases
move "dist\%RELEASE_DIR%" "releases\%RELEASE_DIR%"

set RUN_DIR=releases\%RELEASE_DIR%
(
echo @echo off
echo %FOLDERNAME%.exe --debug
echo pause
) > "%RUN_DIR%\_run_debug.bat"

rmdir /s /q build __pycache__ dist 2>nul
del /s /q *.pyc *.spec *.manifest 2>nul

echo.
echo DONE! %RELEASE_DIR%
echo.
pause
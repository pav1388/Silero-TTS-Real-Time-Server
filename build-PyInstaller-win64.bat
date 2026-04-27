@echo off
setlocal enabledelayedexpansion
set PROGRAMFILE=silero-tts-for-luna-translator.py
set FOLDERNAME=silero-tts-for-luna-translator

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

pyinstaller --onedir --noupx %PROGRAMFILE%

rename dist "%RELEASE_DIR%"
xcopy "models\v5_5_ru.pt" "%RELEASE_DIR%\models\" /I >nul 2>nul
xcopy "README.md" "%RELEASE_DIR%\" >nul 2>nul
xcopy "vitsSimpleAPI_fix\*" "%RELEASE_DIR%\vitsSimpleAPI_fix\" /E /I >nul 2>nul
xcopy "ttsnointerrupt_fix\*" "%RELEASE_DIR%\ttsnointerrupt_fix\" /E /I >nul 2>nul

rmdir /s /q build __pycache__ 2>nul
del /s /q *.pyc *.spec *.manifest 2>nul

echo.
echo DONE! %RELEASE_DIR%
echo.
pause
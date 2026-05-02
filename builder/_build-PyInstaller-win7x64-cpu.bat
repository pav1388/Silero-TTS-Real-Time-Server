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

c:\Python38-64\python.exe -m PyInstaller --onedir --noupx --icon=builder\icon.ico %PROGRAMFILE%

xcopy "models\v5_5_ru.pt" "dist\%FOLDERNAME%\models\" /I >nul 2>nul
xcopy "README.md" "dist\%FOLDERNAME%\" >nul 2>nul
xcopy "tts-rt-server-simple-tester.html" "dist\%FOLDERNAME%\" >nul 2>nul
xcopy "vitsSimpleAPI_fix\*" "dist\%FOLDERNAME%\vitsSimpleAPI_fix\" /E /I >nul 2>nul

rename dist\%FOLDERNAME% "%RELEASE_DIR%"
rename dist releases

rmdir /s /q build __pycache__ 2>nul
del /s /q *.pyc *.spec *.manifest 2>nul

echo.
echo DONE! %RELEASE_DIR%
echo.
pause
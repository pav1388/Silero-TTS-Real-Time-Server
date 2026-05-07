@echo off
title Silero TTS RT Server - Build Script
setlocal enabledelayedexpansion

:: ============================================
:: CONFIGURATION
:: ============================================
set PROGRAMFILE=silero-tts-rt-server.py
set FOLDERNAME=silero-tts-rt-server
set PYTHON_EXE=c:\Python38-64\python.exe
set PATH_7ZIP="c:\TCPU75\Programm\SFX Tool\7zG.exe"
set FALLBACK_7ZIP="C:\Program Files (x86)\7-Zip\7z.exe"
set COMPRESSION_LEVEL=9
set COMPRESSION_METHOD=lzma2:fb=273:d=1024m
:: ============================================

:: Go to parent directory
cd .. 2>nul

:: Detect version
set VERSION=
for /f "tokens=2 delims=^= " %%a in ('findstr "MAIN_VERSION" %PROGRAMFILE% 2^>nul') do (
    if not defined VERSION (
        set VERSION=%%a
        set VERSION=!VERSION:"=!
    )
)
if not defined VERSION set VERSION=0.4
set RELEASE_DIR=%FOLDERNAME%-%VERSION%

:: ============================================
:: ASCII HEADER
:: ============================================
echo.
echo +--------------------------------------------------------------------+
echo ^|                                                                    ^|
echo ^|              SILERO TTS RT SERVER - RELEASE BUILDER                ^|
echo ^|                                                                    ^|
echo +--------------------------------------------------------------------+
echo ^|  Project: %FOLDERNAME%
echo ^|  Version: v%VERSION%
echo +--------------------------------------------------------------------+
echo.

:: Clean old files
echo [1/7] Cleaning old files...
rmdir /s /q build dist __pycache__ 2>nul
del /s /q *.pyc *.spec *.manifest 2>nul
rmdir /s /q "%RELEASE_DIR%" 2>nul
echo        [OK] Cleanup complete
echo.

:: PyInstaller build
echo [2/7] Building with PyInstaller...
%PYTHON_EXE% -m PyInstaller ^
    --onedir ^
    --noupx ^
    --icon=scripts\icon.ico ^
    --exclude-module tensorboard ^
    --exclude-module pytest ^
    %PROGRAMFILE%
if %errorlevel% neq 0 (
    echo        [ERROR] PyInstaller build failed!
    pause
    exit /b 1
)
echo        [OK] Build complete
echo.

:: Copy files
echo [3/7] Copying files...
xcopy "models\v5_5_ru.pt" "dist\%FOLDERNAME%\models\" /I /Y >nul
echo        [OK] Models copied
xcopy "README.md" "dist\%FOLDERNAME%\" /Y >nul
echo        [OK] README.md copied
xcopy "tts-rt-simple-client.html" "dist\%FOLDERNAME%\" /Y >nul
echo        [OK] Tester copied
xcopy "LunaTranslator\*" "dist\%FOLDERNAME%\LunaTranslator\" /E /I /Y >nul
echo        [OK] LunaTranslator copied
echo.

:: Rename and move
echo [4/7] Creating release folder...
rename dist\%FOLDERNAME% "%RELEASE_DIR%"
if not exist releases mkdir releases
move "dist\%RELEASE_DIR%" "releases\%RELEASE_DIR%" >nul
echo        [OK] Release folder: releases\%RELEASE_DIR%
echo.

:: Create debug script
echo [5/7] Creating debug script...
set RUN_DIR=releases\%RELEASE_DIR%
(
echo @echo off
echo title Debug %FOLDERNAME% v%VERSION%
echo echo Starting %FOLDERNAME% in debug mode...
echo echo.
echo %FOLDERNAME%.exe --debug
echo echo.
echo echo Done. Press any key...
echo pause ^>nul
) > "%RUN_DIR%\_run_debug.bat"
echo        [OK] _run_debug.bat created
echo.

:: Create archive
echo [6/7] Creating archive...
if not exist %PATH_7ZIP% (
    if exist %FALLBACK_7ZIP% (
        set PATH_7ZIP=%FALLBACK_7ZIP%
        echo        [INFO] Using default 7-Zip
    ) else (
        echo        [WARNING] 7-Zip not found!
        goto :skip_archive
    )
)

pushd releases >nul
%PATH_7ZIP% a -t7z -ssw -mqs -mx=%COMPRESSION_LEVEL% -myx=%COMPRESSION_LEVEL% -mmt=on -m0=%COMPRESSION_METHOD% -scsWIN "%RELEASE_DIR%.7z" "%RELEASE_DIR%" >nul
popd

if exist "releases\%RELEASE_DIR%.7z" (
    for %%I in ("releases\%RELEASE_DIR%.7z") do set ARCHIVE_SIZE=%%~zI
    set /a ARCHIVE_MB=!ARCHIVE_SIZE!/1048576
    echo        [OK] Archive created: %RELEASE_DIR%.7z ^(!ARCHIVE_MB! MB^)
) else (
    echo        [ERROR] Failed to create archive!
)
:skip_archive
echo.

:: Final cleanup
echo [7/7] Final cleanup...
rmdir /s /q build __pycache__ dist 2>nul
del /s /q *.pyc *.spec *.manifest 2>nul
echo        [OK] Temporary files removed
echo.

:: ============================================
:: FINAL OUTPUT
:: ============================================
echo +--------------------------------------------------------------------+
echo ^|                                                                    ^|
echo ^|   [SUCCESS] BUILD COMPLETED SUCCESSFULLY!                           ^|
echo ^|                                                                    ^|
echo ^|   Release folder: releases\%RELEASE_DIR%
if exist "releases\%RELEASE_DIR%.7z" echo ^|   Archive:        releases\%RELEASE_DIR%.7z
echo ^|                                                                    ^|
echo ^|   To run: %RELEASE_DIR%\%FOLDERNAME%.exe
echo ^|   Debug:  %RELEASE_DIR%\_run_debug.bat
echo ^|                                                                    ^|
echo +--------------------------------------------------------------------+
echo.

echo.
pause
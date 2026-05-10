@echo off
title Silero TTS RT Server - Build Script
setlocal enabledelayedexpansion

:: ============================================
:: CONFIGURATION
:: ============================================
set PROGRAMFILE=silero-tts-rt-server.py
set FOLDERNAME=silero-tts-rt-server
set PYTHON_EXE="C:\Python38-64\python.exe"
set PATH_7ZIP="C:\TCPU75\Programm\SFX Tool\7zG.exe"
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
if not defined VERSION set VERSION=0.0.0
set RELEASE_DIR=%FOLDERNAME%-%VERSION%-win7x64

echo.
echo +--------------------------------------------------------------------+
echo ^|              SILERO TTS RT SERVER - RELEASE BUILDER                ^|
echo +--------------------------------------------------------------------+
echo ^|  Project: %FOLDERNAME%
echo ^|  Version: v%VERSION%
echo +--------------------------------------------------------------------+
echo.

:: ============================================
:: CHECK AND INSTALL DEPENDENCIES
:: ============================================
echo [1/8] Checking Python dependencies...
echo.

set PACKAGES_FAILED=0
set PACKAGES_INSTALLED=0
set PACKAGES_SKIPPED=0

call :InstallPackage "bottle" "bottle" "" "--no-deps"
call :InstallPackage "num2words" "num2words" "" "--no-deps"
call :InstallPackage "typing_extensions" "typing-extensions" "4.5.0" "--no-deps"
call :InstallPackage "mpmath" "mpmath" "1.3.0" "--no-deps"
call :InstallPackage "sympy" "sympy" "1.12" "--no-deps"
call :InstallPackage "numpy" "numpy" "1.24.3" "--no-deps --only-binary :all:"
call :InstallPackage "torch" "torch" "2.0.1" "--no-deps --only-binary :all: --index-url https://download.pytorch.org/whl/cpu"
call :InstallPackage "PyInstaller" "pyinstaller" "" ""

echo.
if %PACKAGES_FAILED% equ 1 (
    echo        [WARNING] %PACKAGES_FAILED% package^(s^) failed to install
    pause
) else (
    echo        [OK] Dependencies ready ^(%PACKAGES_INSTALLED% installed, %PACKAGES_SKIPPED% cached^)
)
echo.
goto :AfterPackages

:InstallPackage
setlocal
set "IMPORT_NAME=%~1"
set "PKG_NAME=%~2"
set "VERSION=%~3"
set "FLAGS=%~4"

%PYTHON_EXE% -c "import %IMPORT_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    endlocal & set /a PACKAGES_SKIPPED+=1
    exit /b 0
)

set "INSTALL_CMD=%PYTHON_EXE% -m pip install --no-cache-dir"
if not "%VERSION%"=="" ( set "INSTALL_CMD=%INSTALL_CMD% %PKG_NAME%==%VERSION%" ) else ( set "INSTALL_CMD=%INSTALL_CMD% %PKG_NAME%" )
if not "%FLAGS%"=="" set "INSTALL_CMD=%INSTALL_CMD% %FLAGS%"

%INSTALL_CMD% >nul 2>&1

if %errorlevel% neq 0 (
    endlocal & set /a PACKAGES_FAILED+=1
    exit /b 1
) else (
    endlocal & set /a PACKAGES_INSTALLED+=1
    exit /b 0
)

:AfterPackages

:: Clean old files
echo [2/8] Cleaning old files...
rmdir /s /q build dist __pycache__ 2>nul
del /s /q *.pyc *.spec *.manifest 2>nul
rmdir /s /q "releases\%RELEASE_DIR%" 2>nul
echo        [OK]
echo.

:: PyInstaller build
echo [3/8] Building with PyInstaller...
%PYTHON_EXE% -m PyInstaller --onedir --noupx --icon=scripts\icon.ico %PROGRAMFILE%

if %errorlevel% neq 0 (
    echo        [ERROR] Build failed!
    pause
    exit /b 1
)
echo        [OK]
echo.

:: Copy files
echo [4/8] Copying files...
xcopy "models\v5_5_ru.pt" "dist\%FOLDERNAME%\models\" /I /Y >nul 2>&1
xcopy "README.md" "dist\%FOLDERNAME%\" /Y >nul 2>&1
xcopy "tts-rt-simple-client.html" "dist\%FOLDERNAME%\" /Y >nul 2>&1
xcopy "LunaTranslator\*" "dist\%FOLDERNAME%\LunaTranslator\" /E /I /Y >nul 2>&1
echo        [OK]
echo.

:: Rename and move
echo [5/8] Creating release folder...
rename dist\%FOLDERNAME% "%RELEASE_DIR%" 2>nul
if not exist releases mkdir releases
move "dist\%RELEASE_DIR%" "releases\%RELEASE_DIR%" >nul 2>&1
echo        [OK] releases\%RELEASE_DIR%
echo.

:: Create debug script
echo [6/8] Creating bat scripts...
set RUN_DIR=releases\%RELEASE_DIR%
(
echo @echo off
echo title %FOLDERNAME% v%VERSION% Debug mode
echo %FOLDERNAME%.exe --debug
echo echo.
echo pause ^>nul
) > "%RUN_DIR%\_run_with_debug.bat"
(
echo @echo off
echo title %FOLDERNAME% v%VERSION% No CPU Monitor mode
echo %FOLDERNAME%.exe --no-cpu-monitor
echo echo.
echo pause ^>nul
) > "%RUN_DIR%\_run_with_no-cpu-monitor.bat"
echo        [OK]
echo.

:: Calculate folder size
echo [7/8] Calculating folder size...
set "FOLDER_PATH=releases\%RELEASE_DIR%"
set FOLDER_SIZE_BYTES=0
for /f "usebackq delims=" %%a in (`powershell -Command "& { (Get-ChildItem -Path '%FOLDER_PATH%' -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum }"`) do set FOLDER_SIZE_BYTES=%%a
if not defined FOLDER_SIZE_BYTES set FOLDER_SIZE_BYTES=0
set /a FOLDER_SIZE_MB=%FOLDER_SIZE_BYTES% / 1048576
echo        [OK] %FOLDER_SIZE_MB% MB
echo.

:: Create archive
echo [8/8] Creating archive...
if not exist %PATH_7ZIP% if exist %FALLBACK_7ZIP% set PATH_7ZIP=%FALLBACK_7ZIP%
if not exist %PATH_7ZIP% (
    echo        [SKIP] 7-Zip not found
    goto :skip_archive
)

pushd releases >nul
%PATH_7ZIP% a -t7z -ssw -mqs -mx=%COMPRESSION_LEVEL% -myx=%COMPRESSION_LEVEL% -mmt=on -m0=%COMPRESSION_METHOD% -scsWIN "%RELEASE_DIR%.7z" "%RELEASE_DIR%" >nul 2>&1
popd

if exist "releases\%RELEASE_DIR%.7z" (
    for %%I in ("releases\%RELEASE_DIR%.7z") do set ARCHIVE_SIZE=%%~zI
    set /a ARCHIVE_MB=!ARCHIVE_SIZE!/1048576
    echo        [OK] !ARCHIVE_MB! MB
) else (
    echo        [ERROR] Archive creation failed
)
:skip_archive
echo.

:: Final cleanup
echo [*] Final cleanup...
rmdir /s /q build __pycache__ dist 2>nul
del /s /q *.pyc *.spec *.manifest 2>nul
echo        [OK]
echo.

:: Final output
echo +--------------------------------------------------------------------+
echo ^|   [SUCCESS] BUILD COMPLETED                                        ^|
echo +--------------------------------------------------------------------+
echo ^|   Release: releases\%RELEASE_DIR% (%FOLDER_SIZE_MB% MB^)
if exist "releases\%RELEASE_DIR%.7z" echo ^|   Archive: releases\%RELEASE_DIR%.7z (!ARCHIVE_MB! MB^)
echo +--------------------------------------------------------------------+
echo.
pause
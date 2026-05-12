@echo off
title Silero TTS RT Server - Build Script
setlocal enabledelayedexpansion

:: ================= CONFIG =================
set "PROGRAMFILE=silero-tts-rt-server.py"
set "FOLDERNAME=silero-tts-rt-server"
set "PYTHON_EXE=C:\Python38-64\python.exe"
set "SEVENZIP_EXE=C:\TCPU75\Programm\SFX Tool\7z.exe"
set "FALLBACK_7ZIP=C:\Program Files (x86)\7-Zip\7z.exe"
set "PRE_ARCHIVE_NAME=pre-archive.7z"
set "PRE_ARCHIVE_PATH=scripts\%PRE_ARCHIVE_NAME%"
:: =========================================

:: TIMER START
for /f "tokens=1-3 delims=:." %%a in ("%TIME%") do (
    set /a START_H=1%%a-100
    set /a START_M=1%%b-100
    set /a START_S=1%%c-100
)
set /a START_TIME=START_H*3600 + START_M*60 + START_S

:: Go to parent directory
cd .. 2>nul

:: VERSION DETECT
set VERSION=
for /f "tokens=2 delims=^= " %%a in ('findstr "MAIN_VERSION" %PROGRAMFILE% 2^>nul') do (
    if not defined VERSION (
        set "VERSION=%%a"
        set "VERSION=!VERSION:"=!"
    )
)
if not defined VERSION set VERSION=0.0.0

set "RELEASE_NAME=%FOLDERNAME%-%VERSION%-win7x64"

echo.
echo ============================================
echo     SILERO TTS RT SERVER - BUILD SCRIPT
echo ============================================
echo   Version: %VERSION%
echo ============================================
echo.

:: ================= DEPENDENCIES =================
echo [1/7] Python dependencies...

call :pip_install bottle
call :pip_install num2words
call :pip_install typing-extensions 4.5.0
call :pip_install mpmath 1.3.0
call :pip_install sympy 1.12
call :pip_install numpy 1.24.3 --no-deps --only-binary :all:
call :pip_install torch 2.0.1 --no-deps --only-binary :all: --index-url https://download.pytorch.org/whl/cpu
call :pip_install pyinstaller

echo.

:: ================= CLEAN =================
echo [2/7] Cleaning...
rmdir /s /q build dist __pycache__ 2>nul
del /s /q *.pyc *.spec *.manifest 2>nul
rmdir /s /q "releases\%FOLDERNAME%" 2>nul
echo        [OK]

:: ================= BUILD =================
echo [3/7] PyInstaller build...

%PYTHON_EXE% -m PyInstaller --onedir --noupx --icon=scripts\icon.ico %PROGRAMFILE%

if not errorlevel 0 (
    echo BUILD FAILED
    pause
    exit /b 1
)
echo        [OK]

:: ================= RELEASE =================
echo [4/7] Creating release...

if not exist releases mkdir releases
move "dist\%FOLDERNAME%" "releases\%FOLDERNAME%" >nul 2>&1
rmdir /s /q build dist __pycache__ 2>nul
del /s /q *.pyc *.spec *.manifest 2>nul

echo        [OK]

:: ================= COPY FILES =================
echo [5/7] Copy files...

xcopy "models\v5_5_ru.pt" "releases\%FOLDERNAME%\models\" /I /Y >nul 2>&1
xcopy "README.md" "releases\%FOLDERNAME%\" /Y >nul 2>&1
xcopy "tts-rt-simple-client.html" "releases\%FOLDERNAME%\" /Y >nul 2>&1
xcopy "LunaTranslator\*" "releases\%FOLDERNAME%\LunaTranslator\" /E /I /Y >nul 2>&1

echo        [OK]

:: ================= CREATE SCRIPTS =================
echo [6/7] Create bat scripts...

set RUN_DIR=releases\%FOLDERNAME%
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

:: ================= CREATE ARCHIVE =================
echo.
echo [7/7] Creating archive...

if not exist "%SEVENZIP_EXE%" if exist "%FALLBACK_7ZIP%" set "SEVENZIP_EXE=%FALLBACK_7ZIP%"

if not exist "%SEVENZIP_EXE%" (
    echo 7-Zip not found, skipping archive
    goto :finish
)

:: PRE ARCHIVE MODE
if exist "%PRE_ARCHIVE_PATH%" (
    echo Using pre-archive mode...

    copy "%PRE_ARCHIVE_PATH%" "releases\%PRE_ARCHIVE_NAME%" >nul 2>&1
    pushd releases >nul
    echo Generating file list from %PRE_ARCHIVE_NAME%...
    del archive_files_raw.txt 2>nul
    del excludes.txt 2>nul
    "%SEVENZIP_EXE%" l -ba "%PRE_ARCHIVE_NAME%" > archive_files_raw.txt
    > "excludes.txt" (
        for /f "tokens=5,6*" %%a in (archive_files_raw.txt) do (
            set "size=%%a"
            set "size=!size:,=!"
            if not "!size!"=="0" (
                if not "!size!"=="" (
                    echo %%b
                )
            )
        )
    )
    del archive_files_raw.txt 2>nul
    
    :: Подсчитываем количество исключаемых файлов
    for /f %%c in ('type "excludes.txt" 2^>nul ^| find /c /v ""') do set "COUNT=%%c"
    echo Excluding !COUNT! files from update.
    
    "%SEVENZIP_EXE%" u "%PRE_ARCHIVE_NAME%" "%FOLDERNAME%" ^
        -t7z -ms=off -ssw -bsp1 -mx=9 -myx=9 -mmt=on -m0=lzma2:fb=273:d=1024m ^
        -x@excludes.txt 2>nul
    
    del excludes.txt 2>nul
    rename "%PRE_ARCHIVE_NAME%" "%RELEASE_NAME%.7z" >nul 2>&1

    popd
    goto :archive_done
)

:: FULL ARCHIVE MODE
echo Using full archive mode...
pushd releases >nul

"%SEVENZIP_EXE%" a "%FOLDERNAME%.7z" "%FOLDERNAME%" ^
    -t7z -ms=off -ssw -bsp1 -mx=9 -myx=9 -mmt=on -m0=lzma2:fb=273:d=1024m 2>nul

rename "%FOLDERNAME%.7z" "%RELEASE_NAME%.7z" >nul 2>&1

popd

:archive_done

:: ================= TIMER END =================
:finish

for /f "tokens=1-3 delims=:." %%a in ("%TIME%") do (
    set /a END_H=1%%a-100
    set /a END_M=1%%b-100
    set /a END_S=1%%c-100
)

set /a END_TIME=END_H*3600 + END_M*60 + END_S
set /a ELAPSED=END_TIME-START_TIME
if !ELAPSED! lss 0 set /a ELAPSED+=86400

set /a MIN=ELAPSED/60
set /a SEC=ELAPSED%%60

echo.
echo ============================================
echo BUILD COMPLETE
echo ============================================
echo Release: releases\%FOLDERNAME%
if exist "releases\%RELEASE_NAME%.7z" echo Archive: releases\%RELEASE_NAME%.7z
echo Time: !MIN!m !SEC!s
echo ============================================
echo.

pause
exit /b


:: ================= FUNCTIONS =================
:pip_install
set "PKG=%~1"
set "VER=%~2"

if "%VER%"=="" (
    "%PYTHON_EXE%" -c "import %PKG%" >nul 2>&1
    if not errorlevel 1 exit /b
    "%PYTHON_EXE%" -m pip install %PKG%
) else (
    "%PYTHON_EXE%" -c "import %PKG%" >nul 2>&1
    if not errorlevel 1 exit /b
    "%PYTHON_EXE%" -m pip install %PKG%==%VER% %3 %4 %5 %6
)

exit /b
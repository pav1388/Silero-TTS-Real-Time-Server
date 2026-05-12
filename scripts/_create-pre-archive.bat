@echo off
set "FOLDERNAME=silero-tts-rt-server"
set "SEVENZIP_EXE=C:\TCPU75\Programm\SFX Tool\7z.exe"
set "FALLBACK_7ZIP=C:\Program Files (x86)\7-Zip\7z.exe"

cd ..
pushd releases >nul

"%SEVENZIP_EXE%" a -t7z -ssw -ms=off -mqs -mx=9 -myx=9 -mmt=2 -m0=lzma2:fb=273:d=1024m "pre-archive.7z" "%FOLDERNAME%" -scsWIN

popd
move "releases\pre-archive.7z" "scripts\pre-archive.7z" >nul 2>&1

pause

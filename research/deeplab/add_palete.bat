@echo off

:: Please inform the color palete you want to use here:
set Pal=RUGD_vsetky_farby.pal

:: Please inform the path to irfranview here:
set Irfran=c:\Program Files\IrfanView\i_view64.exe 

if exist "%~1\" (set "Folder=%~1") else (exit)

pushd "%Folder%"

for /f "delims=" %%a in ('dir /b *.jpg *.png *.bmp') do "%Irfran%" "%%a" /import_pal="%Pal%" /convert="%%~na_new%%~xa"
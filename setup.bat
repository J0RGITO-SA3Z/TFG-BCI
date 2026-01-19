@echo off
echo Iniciando instalacion del proyecto...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

echo.
echo Pulsa una tecla para cerrar...
pause > nul
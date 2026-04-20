$envName = "bci-mi-tfg"

Set-Location $PSScriptRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Conda no está disponible en el PATH"
    Read-Host "Pulsa Enter para salir"
    exit 1
}

conda shell.powershell hook | Out-String | Invoke-Expression

if (-not (conda env list | Select-String $envName)) {
    Write-Host "❌ El entorno '$envName' no existe"
    Read-Host "Pulsa Enter para salir"
    exit 1
}

conda activate $envName

Write-Host ""
Write-Host "  ¿Qué quieres ejecutar?"
Write-Host "  [1] Aplicación EEG principal"
Write-Host "  [2] Visualizador en tiempo real (otro portátil)"
Write-Host ""

$opcion = Read-Host "Elige (1/2)"

switch ($opcion) {
    "1" { python EEG_app\src\app\aplicacion.py }
    "2" { python -m EEG_app.src.visualizer_app.main }
    default {
        Write-Host "❌ Opción no válida. Escribe 1 o 2."
        Read-Host "Pulsa Enter para salir"
        exit 1
    }
}

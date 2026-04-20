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
python EEG_app\src\app\aplicacion.py

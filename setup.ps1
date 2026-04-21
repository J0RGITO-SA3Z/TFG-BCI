$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SETUP DEL PROYECTO MIREPNET (TFG)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# -----------------------------
# Variables
# -----------------------------
$BrainAccessUrl = "https://www.brainaccess.ai/wp-content/uploads/downloads/BrainAccessSDK-classic.zip"
$zipFile = "BrainAccess.zip"
$extractDir = "lib/BrainAccess"
$envName = "bci-mi-tfg"
$envFile = "./bci-mi-tfg.yml"

# -----------------------------
# Comprobar Conda
# -----------------------------
Write-Host "[1/5] Comprobando Conda..." -ForegroundColor Cyan

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error @"
Conda no esta instalado o no esta en el PATH.

Instala Anaconda y vuelve a ejecutar este script:
https://www.anaconda.com/download
"@
    exit 1
}

Write-Host "[OK] Conda detectado" -ForegroundColor Green
Write-Host ""

# -----------------------------
# Crear entorno Conda
# -----------------------------
Write-Host "[2/5] Creando entorno Conda..." -ForegroundColor Cyan

if (conda env list | Select-String $envName) {
    Write-Host "El entorno '$envName' ya existe, eliminando para recrearlo desde cero..." -ForegroundColor Yellow
    conda env remove -n $envName -y
    Write-Host "[OK] Entorno anterior eliminado" -ForegroundColor Green
}

conda env create -f $envFile -n $envName
Write-Host "[OK] Entorno creado" -ForegroundColor Green

Write-Host ""

# -----------------------------
# Activar entorno Conda
# -----------------------------
Write-Host "[3/5] Activando entorno Conda..." -ForegroundColor Cyan

& conda shell.powershell hook | Out-String | Invoke-Expression
conda activate $envName

Write-Host "[OK] Entorno activado" -ForegroundColor Green
Write-Host ""

# -----------------------------
# Instalar BrainAccess SDK
# -----------------------------
Write-Host "[4/5] Instalando BrainAccess SDK..." -ForegroundColor Cyan

if (-not (Test-Path $extractDir)) {

    Write-Host "Descargando BrainAccess SDK..." -ForegroundColor Cyan
    Invoke-WebRequest $BrainAccessUrl -OutFile $zipFile

    Write-Host "Descomprimiendo BrainAccess SDK..." -ForegroundColor Cyan
    Expand-Archive -Path $zipFile -DestinationPath $extractDir -Force
    Remove-Item $zipFile

} else {
    Write-Host "Carpeta BrainAccess ya existe, se omite la descarga" -ForegroundColor Yellow
}

$pythonApiPath = Join-Path $extractDir "PythonAPI"

if (-not (Test-Path $pythonApiPath)) {
    Write-Error "No se encontro la carpeta PythonAPI"
    exit 1
}

Write-Host "Instalando BrainAccess en el entorno activo..." -ForegroundColor Cyan
Push-Location $pythonApiPath
pip install .
Pop-Location

Write-Host "[OK] BrainAccess SDK instalado" -ForegroundColor Green

Write-Host ""

# -----------------------------
# PATH (opcional)
# -----------------------------
Write-Host "[5/5] Configurando PATH del usuario..." -ForegroundColor Cyan

$libPath = (Resolve-Path $extractDir).Path
$oldPath = [Environment]::GetEnvironmentVariable("PATH", "User")

if ($oldPath -notlike "*$libPath*") {
    [Environment]::SetEnvironmentVariable(
        "PATH",
        "$oldPath;$libPath",
        "User"
    )
    Write-Host "[OK] PATH actualizado (reinicia la terminal)" -ForegroundColor Yellow
} else {
    Write-Host "El PATH ya contenia la ruta necesaria" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETADO CORRECTAMENTE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
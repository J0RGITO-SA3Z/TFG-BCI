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
$modelsDir = "lib/Modelos"
$envName = "mirepnet_env"
$envFile = "./mirepnet_env.yml"

# -----------------------------
# Comprobar Conda
# -----------------------------
Write-Host "[1/6] Comprobando Conda..." -ForegroundColor Cyan

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
# Clonar modelo
# -----------------------------
Write-Host "[2/6] Preparando carpeta de modelos..." -ForegroundColor Cyan

New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

Push-Location $modelsDir

if (-not (Test-Path "MIRepNet")) {
    Write-Host "Clonando repositorio MIRepNet..." -ForegroundColor Cyan
    git clone https://github.com/staraink/MIRepNet
    Write-Host "[OK] Repositorio clonado" -ForegroundColor Green
} else {
    Write-Host "MIRepNet ya existe, se omite el clon" -ForegroundColor Yellow
}

Pop-Location
Write-Host ""

# -----------------------------
# Crear entorno Conda
# -----------------------------
Write-Host "[3/6] Creando entorno Conda..." -ForegroundColor Cyan

if (-not (conda env list | Select-String $envName)) {
    conda env create -f $envFile -n $envName
    Write-Host "[OK] Entorno creado" -ForegroundColor Green
} else {
    Write-Host "El entorno ya existe, se omite la creacion" -ForegroundColor Yellow
}

Write-Host ""

# -----------------------------
# Activar entorno Conda
# -----------------------------
Write-Host "[4/6] Activando entorno Conda..." -ForegroundColor Cyan

& conda shell.powershell hook | Out-String | Invoke-Expression
conda activate $envName

Write-Host "[OK] Entorno activado" -ForegroundColor Green
Write-Host ""

# -----------------------------
# Instalar BrainAccess SDK
# -----------------------------
Write-Host "[5/6] Instalando BrainAccess SDK..." -ForegroundColor Cyan

if (-not (Test-Path $extractDir)) {

    Write-Host "Descargando BrainAccess SDK..." -ForegroundColor Cyan
    Invoke-WebRequest $BrainAccessUrl -OutFile $zipFile

    Write-Host "Descomprimiendo BrainAccess SDK..." -ForegroundColor Cyan
    Expand-Archive -Path $zipFile -DestinationPath $extractDir -Force
    Remove-Item $zipFile

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
} else {
    Write-Host "BrainAccess ya esta instalado, se omite" -ForegroundColor Yellow
}

Write-Host ""

# -----------------------------
# PATH (opcional)
# -----------------------------
Write-Host "[6/6] Configurando PATH del usuario..." -ForegroundColor Cyan

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
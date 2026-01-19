$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  SETUP DEL PROYECTO MIREPNET (TFG)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$BrainAccessUrl = "https://www.brainaccess.ai/wp-content/uploads/downloads/BrainAccessSDK-classic.zip"
$zipFile = "BrainAccess.zip"
$extractDir = "BrainAccess"

$modelsDir = "Modelos"

# --------------------------------------------------
# Comprobación de Conda
# --------------------------------------------------
Write-Host "[1/6] Comprobando instalación de Conda..." -ForegroundColor Cyan

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error @"
Conda no está instalado o no está en el PATH.

Instala Anaconda y vuelve a ejecutar este script:
https://www.anaconda.com/download
"@
    exit 1
}

Write-Host "✔ Conda detectado correctamente" -ForegroundColor Green
Write-Host ""

# --------------------------------------------------
# Clonado del modelo
# --------------------------------------------------
Write-Host "[2/6] Preparando carpeta de modelos..." -ForegroundColor Cyan

if (-not (Test-Path $modelsDir)) {
    Write-Host "Creando carpeta '$modelsDir'..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

Push-Location $modelsDir

if (-not (Test-Path "MIRepNet")) {
    Write-Host "Clonando repositorio MIRepNet..." -ForegroundColor Cyan
    git clone https://github.com/staraink/MIRepNet
    Write-Host "✔ Repositorio clonado" -ForegroundColor Green
} else {
    Write-Host "El repositorio MIRepNet ya existe. Se omite el clon." -ForegroundColor Yellow
}

Pop-Location
Write-Host ""

# --------------------------------------------------
# Creación del entorno Conda
# --------------------------------------------------
Write-Host "[3/6] Creando entorno Conda..." -ForegroundColor Cyan

conda env create -f ./env/mirepnet_env.yml

Write-Host "✔ Entorno Conda creado" -ForegroundColor Green
Write-Host ""

# --------------------------------------------------
# Activación del entorno
# --------------------------------------------------
Write-Host "[4/6] Activando entorno Conda 'mirepnet_env'..." -ForegroundColor Cyan

& conda shell.powershell hook | Out-String | Invoke-Expression
conda activate mirepnet_env

Write-Host "✔ Entorno activado correctamente" -ForegroundColor Green
Write-Host ""

# --------------------------------------------------
# Instalación de BrainAccess SDK
# --------------------------------------------------
Write-Host "[5/6] Descargando BrainAccess SDK..." -ForegroundColor Cyan

Invoke-WebRequest $BrainAccessUrl -OutFile $zipFile

Write-Host "Descomprimiendo BrainAccess SDK..." -ForegroundColor Cyan
Expand-Archive -Path $zipFile -DestinationPath $extractDir -Force
Remove-Item $zipFile

$pythonApiPath = Join-Path $extractDir "PythonAPI"

Write-Host "Instalando BrainAccess SDK en el entorno activo..." -ForegroundColor Cyan
Push-Location $pythonApiPath
pip install .
Pop-Location

Write-Host "✔ BrainAccess SDK instalado correctamente" -ForegroundColor Green
Write-Host ""

# --------------------------------------------------
# PATH (si aplica)
# --------------------------------------------------
Write-Host "[6/6] Configurando PATH del usuario..." -ForegroundColor Cyan

if ($oldPath -notlike "*$libPath*") {
    [Environment]::SetEnvironmentVariable(
        "PATH",
        "$oldPath;$libPath",
        "User"
    )
    Write-Host "✔ PATH actualizado (reinicia la terminal)" -ForegroundColor Yellow
} else {
    Write-Host "El PATH ya contenía la ruta necesaria" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETADO CORRECTAMENTE" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

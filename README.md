# TFG-BCI: Interfaz Cerebro-Computador para Videojuegos

Sistema BCI basado en imaginación motora que clasifica señales EEG en tiempo real para controlar videojuegos y parámetros MIDI. Desarrollado como Trabajo de Fin de Grado.

> **Importante:** La aplicación está diseñada exclusivamente para el dispositivo **BrainAccess Mini**. No es compatible con otros cascos EEG.

---

## Índice

1. [Descripción del proyecto](#1-descripción-del-proyecto)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Dataset](#3-dataset)
4. [Puesta en marcha](#4-puesta-en-marcha)
5. [Aplicación principal](#5-aplicación-principal)
6. [Visualizador TCP](#6-visualizador-tcp)
7. [Dependencias](#7-dependencias)

---

## 1. Descripción del proyecto

Este proyecto desarrolla una **interfaz cerebro-computador (BCI) para jugar videojuegos** mediante imaginación motora. El usuario piensa en movimientos (mano izquierda, mano derecha, ambas manos, pies, reposo) y el sistema clasifica esas señales cerebrales en tiempo real para traducirlas en acciones de juego o mensajes MIDI.

El pipeline completo incluye:

- Adquisición de EEG con el casco **BrainAccess Mini** (15 canales)
- Preprocesado de señal: filtrado paso banda, eliminación de artefactos, re-referenciado CAR, interpolación de canales malos
- Clasificación con **MiRepNet**, un modelo de deep learning preentrenado para imaginación motora
- Filtros de decisión para suavizar las predicciones en tiempo real (votación por mayoría, suavizado exponencial, integrador con fuga, etc.)
- Control de videojuegos (Flappy Bird, Arrow Runner, Endless Wave Runner) y dispositivos MIDI

---

## 2. Estructura del repositorio

```
TFG-BCI/
├── EEG_app/                  # Aplicación principal y toda la lógica del sistema
│   ├── src/
│   │   ├── app/              # GUI principal (menú, grabación, experimentos, juegos)
│   │   ├── components/       # Pipeline de procesado, modelos, filtros de decisión
│   │   ├── visualizer_app/   # Visualizador remoto via TCP
│   │   ├── offline_tests/    # Scripts de análisis y validación offline
│   │   └── util/             # Utilidades varias
│   ├── recordings/           # Dataset EEG grabado con el casco
│   │   ├── experimento_visual/   # 8 sujetos, paradigma visual
│   │   ├── experimento_juegos/   # 3 sujetos, experimento con videojuegos
│   │   ├── piloto/               # 4 sujetos, estudio piloto
│   │   └── simulations_RT/       # Simulaciones de filtros de decisión
│   ├── results/              # Gráficas de resultados de experimentos
│   └── config/               # Configuraciones de canales y acciones
├── data/                     # Datasets externos (BCI Competition IV)
├── Memoria/                  # Documentación del TFG en LaTeX
├── lib/                      # BrainAccess SDK (se descarga con setup.bat)
├── setup.bat                 # Instalación del entorno (ejecutar una vez)
├── run.bat                   # Lanzador de la aplicación
└── bci-mi-tfg.yml            # Especificación del entorno Conda
```

### EEG_app

Contiene todo el sistema BCI. El núcleo está en `src/components/`, que implementa el pipeline modular de procesado: proveedores de datos, preprocesado de señal raw, segmentación en épocas, detección de canales malos, interfaz con MiRepNet y filtros de decisión. La GUI en `src/app/` expone todas las funcionalidades a través de un menú de consola.

### Memoria

Documentación completa del TFG escrita en LaTeX. Incluye el estado del arte, la descripción del sistema, los experimentos realizados, resultados y conclusiones, además de los diagramas fuente en formato Draw.io.

---

## 3. Dataset

El repositorio incluye un dataset propio grabado con el casco BrainAccess Mini en `EEG_app/recordings/experimento_visual/`:

- **8 sujetos**, 4 sesiones cada uno (32 sesiones en total)
- Paradigma de **imaginación motora con señales visuales**
- Formato MNE FIF (`sujN_1-4_raw.fif`), 15 canales
- Metadatos por sujeto en `sujN_perfil.json` (datos demográficos e impedancias por sesión)

Adicionalmente se dispone del dataset público **BCI Competition IV** (9 sujetos) en `data/BCICIV_2a_gdf/`, usado para validación del modelo.

---

## 4. Puesta en marcha

### Requisitos previos

- Windows 10/11
- [Anaconda](https://www.anaconda.com/) instalado
- GPU NVIDIA con CUDA 11.8 para la inferencia del modelo

### Instalación (solo la primera vez)

Ejecuta el script de configuración desde la raíz del repositorio:

```
setup.bat
```

Esto crea el entorno Conda `bci-mi-tfg` con Python 3.10, PyTorch 2.2.0 y todas las dependencias, y descarga el SDK de BrainAccess.

### Ejecutar la aplicación

```
run.bat
```

Muestra un menú para elegir entre la aplicación principal o el visualizador remoto.

---

## 5. Aplicación principal

Lanzada desde `run.bat`, ofrece un menú con las siguientes opciones:

| Opción | Descripción |
|--------|-------------|
| Configuración de canales | Seleccionar qué electrodos usar |
| Configuración de acciones | Mapear clases de imaginación motora |
| Estado de batería | Ver nivel de batería del casco |
| Medición de impedancias | Comprobar contacto de los electrodos |
| Grabación manual | Registrar señal EEG libremente |
| Visualización en directo | Ver señal en tiempo real |
| Experimento visual | Protocolo de grabación con señales visuales |
| Evaluación de sujeto | Clasificación offline con MiRepNet |
| Experimento en tiempo real | Jugar con el BCI (Flappy Bird, Arrow Runner...) |
| Ver archivos FIF | Explorar grabaciones almacenadas |

---

## 6. Visualizador TCP

Aplicación independiente que permite monitorizar la señal EEG en tiempo real desde **otro ordenador** a través de la red. Se conecta al servidor TCP que lanza la aplicación principal (`src/tcp/eeg_live_server.py`).

Para ejecutarlo directamente:

```bash
python -m EEG_app.src.visualizer_app.main
```

---

## 7. Dependencias principales

| Categoría | Librería |
|-----------|----------|
| Señal / ML | MNE 1.6.1, PyTorch 2.2.0, scikit-learn 1.3.2, scipy 1.15.1 |
| Datos | NumPy 1.26.4, Pandas 1.5.3, MOABB 1.0.0 |
| Hardware | BrainAccess SDK |
| GUI / Juegos | PyQt5 5.15.11, PyGame 2.6.1, pyqtgraph 0.14.0 |
| Monitorización | Rich 14.3.1, wandb 0.22.2 |

El entorno completo está definido en `bci-mi-tfg.yml` y se instala automáticamente con `setup.bat`.

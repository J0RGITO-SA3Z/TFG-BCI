# Poner en marcha el modelo MiRepNet

---
## 1️⃣ Desde Anaconda Prompt, dirígete al directorio /env y ejecuta:
```bash
conda env create -f mirepnet_env.yml
```
---
## 2️⃣ Clonar el repositorio del modelo, en el directorio Modelos, ejecuta:
```bash
git clone https://github.com/yourusername/MIRepNet.git
```
---
## 3️⃣ (Opcional) Instalar CUDA para aceleración GPU
Si tu equipo cuenta con una tarjeta gráfica NVIDIA, instala CUDA para mejorar significativamente el rendimiento.

---
## 4️⃣ Abrir el proyecto en VS con entorno virtual -> ctrl +shift + p  -> Python select interpreter -> mirepnet_env

TFG-BCI/
│
├── data/
│ ├── Grabaciones casco/
│ │ - Archivos .db originales del casco
│ │ - .fif, .npy, .csv procesados
│ │ - Anotaciones y sesiones
│ └── BCICIV_2a_gdf/
│ - Dataset público para validación
│
├── env/
│ └── mirepnet_env.yml
│ - Entorno Conda para MIRepNet + MNE + PyTorch
│
├── src/
│ ├── brainaccess/
│ │ └── brainaccess.py
│ │ - Lectura del casco en tiempo real
│ │
│ ├── preprocessing/
│ │ - (Filtros, alineación, proyectores, normalización...)
│ │
│ ├── mirepnet/
│ │ ├── mirepnetest.py
│ │ │ - Test de lectura del casco + inferencia MIRepNet
│ │ └── tests/
│ │ - Scripts de prueba y experimentos
│ │
│ ├── utils/
│ │ ├── distance_2d.py
│ │ ├── distance_3d.py
│ │ - Cálculo de distancias Euclídeas 2D/3D entre plantillas
│ │
│ └── pipelines/
│ ├── Conversor.py
│ │ - Conversor .db → .fif, .csv, .npy
│ └── Grabaciones.py
│ - Pruebas de lectura de archivos .db (probablemente se eliminará)
│
├── Modelos/
│ └── MIRepNet/
│ - Repositorio clonado como submódulo (modelo + pesos)
│
└── README.md
 📜 mirepnet_env.yml      ← Archivo del entorno Conda
---

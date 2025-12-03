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
│   ├── Grabaciones casco/# .db originales del casco , .fif, .npy, .csv limpios, anotaciones y sesiones
|   └──  BCICIV_2a_gdf
│   
├── env/
│   ├──  mirepnet_env.yml 
│
├── src/
│   ├── brainaccess/
│   │   └── brainaccess.py   # Lectura del casco en tiempor real
│   │
│   ├── preprocessing/
│   │
│   ├── mirepnet/
│   │   ├── mirepnetest.py   # Test lectura de casco + MIRepNet
│   │   └── tests/    # carpeta de srcipts de prueba
│   │
│   ├── utils/
│   │   ├── distance_2d.py
│   │   ├── distance_3d.py
│   │
│   └── pipelines/
│       ├── Conversor.py  # .db → MNE/CSV/fif/npy
│       └── Grabaciones.py      # pruebas de ñectura archivos db (lo borraremos seguramente)
│
│
├── Modelos/
│   └── MIRepNet/     # tu repo clonado como submódulo
│
└── README.md

 
 📜 mirepnet_env.yml      ← Archivo del entorno Conda
---

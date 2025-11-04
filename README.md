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
## 4️⃣ Abrir el proyecto en PyCharm
Dentro del directorio scripts/MIRepNet encontrarás una carpeta llamada EEG.
Esta carpeta contiene el proyecto de PyCharm, que puedes abrir y ejecutar asegurándote de que esté seleccionado el entorno virtual mirepnet_env.

📦 Modelos/

 ┣ 📂 MIRepNet/
 
 ┃ ┣ 📂 scripts/
 
 ┃ ┃ ┣ 📂 EEG/   ← Proyecto PyCharm
 
 ┃ ┃ ┣ 📜 test_mirepnet.py
 
 ┃ ┃ ┗ 📜 train_mirepnet.py
 
 📜 mirepnet_env.yml      ← Archivo del entorno Conda
---

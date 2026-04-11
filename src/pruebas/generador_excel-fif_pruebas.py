import mne
import numpy as np
import pandas as pd

def generar_datos_prueba():
    print("Generando archivos de prueba...")

    # ==========================================
    # 1. GENERAR ARCHIVO FIF (falso EEG)
    # ==========================================
    sfreq = 250  # Frecuencia de muestreo (Hz)
    n_channels = 8 # Número de canales
    duracion_segundos = 60

    # Crear información de canales y datos aleatorios (ruido)
    info = mne.create_info(ch_names=[f'EEG{i}' for i in range(n_channels)], sfreq=sfreq, ch_types='eeg')
    data = np.random.randn(n_channels, sfreq * duracion_segundos)
    raw = mne.io.RawArray(data, info)

    # Crear 20 anotaciones (eventos) espaciadas cada 2 segundos
    onsets = np.arange(2, 42, 2)
    durations = [1.0] * len(onsets)
    
    # Alternar entre izquierda y derecha
    true_labels = ['left_hand' if i % 2 == 0 else 'right_hand' for i in range(len(onsets))]

    annotations = mne.Annotations(onset=onsets, duration=durations, description=true_labels)
    raw.set_annotations(annotations)
    
    # Guardar FIF
    fif_name = "fif_origin.fif"
    raw.save(fif_name, overwrite=True)
    print(f"✅ Creado: {fif_name} con {len(onsets)} eventos.")

    # ==========================================
    # 2. GENERAR ARCHIVO EXCEL (falsas predicciones)
    # ==========================================
    predicciones = []
    probabilidades = []

    # Simulamos que el modelo acierta el 80% de las veces
    for label in true_labels:
        if np.random.rand() > 0.2:
            # Acierto: ponemos la etiqueta correcta y una probabilidad alta
            predicciones.append(label)
            probabilidades.append(np.random.uniform(0.70, 0.95))
        else:
            # Fallo: ponemos la etiqueta contraria y una probabilidad más baja
            error_label = 'right_hand' if label == 'left_hand' else 'left_hand'
            predicciones.append(error_label)
            probabilidades.append(np.random.uniform(0.51, 0.65))

    # Crear DataFrame
    df = pd.DataFrame({
        'prediction': predicciones,
        'probs': probabilidades,
        'sample': onsets * sfreq, # Muestra aproximada
        'delay': [0.015] * len(onsets) # Delay falso
    })

    # Guardar Excel
    excel_name = "excel_prueba.xlsx"
    df.to_excel(excel_name, index=False)
    print(f"✅ Creado: {excel_name} con {len(predicciones)} predicciones.")

if __name__ == "__main__":
    # Fijamos la semilla para que siempre salgan los mismos datos aleatorios
    np.random.seed(42) 
    generar_datos_prueba()
    print("\n¡Listo! Ahora puedes ejecutar tu script offline apuntando a estos dos archivos.")
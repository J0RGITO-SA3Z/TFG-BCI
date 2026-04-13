import mne
import os, sys

# Ensure project `src` folder is on sys.path so top-level imports like
# `DataProvider.FifDataProvider` work when running this file directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from simulacion_tiempo_real.trainingOffline import Training_offline
import os

from simulacion_tiempo_real.RT_modules.RT_pipeline_process import RT_pipeline_process
from simulacion_tiempo_real.RT_modules.real_time_emulator import RealTimeEmulator
from simulacion_tiempo_real.EEGSimulator import EEGSimulator


from simulacion_tiempo_real.RT_interpreters.RT_interpreter_process import RT_interpreter_process

import time
from rich.console import Console

def experimento_offline_RT(emulationfif,Trainingfif, SalidaPredicciones, console):


    # Emulacion de entrenamiento offline para obtener matriz de características y modelo entrenado.
    rtTraining = Training_offline()
    EA_matrix, model = rtTraining.start(Trainingfif, lista=["left_hand", "right_hand"], epochs=10, seed=42, validation_split=0.2)

    # En lugar de conectar a un dispositivo EEG real, usamos un emulador que lee los epochs
    # del `FifDataProvider` y los va enviando como si fueran datos en tiempo real.
    # (usar una única instancia; evitar pasar objetos incorrectos a mne.read_raw_fif)
    eeg = EEGSimulator(emulationfif)
    interpreter_process = None
    modelPipeline = None

    # Crea el interprete en proceso separado para recibir predicciones.
    interpreter_process = RT_interpreter_process()
    interpreter_process.start(filename=SalidaPredicciones)

    # Crea el pipeline en proceso separado y conecta su salida al interprete.
    modelPipeline = RT_pipeline_process(
        ea_matrix = EA_matrix,
        pretrained_model = model,
        info = eeg.get_info(),
        channel_positions = eeg.get_channel_indexes()
    )
    modelPipeline.run_process(interpreter_process)

    eeg.register_callback(modelPipeline.sendData)
    console.print("Cargando buffer inicial durante 15 segundos antes de activar predicciones...")
    eeg.iniciarGrabacion(15)
    console.print("Buffer cargado.")
    modelPipeline.activar_predecir()
    eeg.iniciarGrabacion()

    modelPipeline.stop_process()
    interpreter_process.stop()

    print("grabacion terminada")


def interfaz_experimento_RT(emulation_fif, fif_train, console):
    console.print("=== Experimento de Simulación en Tiempo Real ===")
    nombre_experimento = ""
    while not nombre_experimento:
        nombre_experimento = input("Nombre del experimento (se creara una carpeta en recordings): ").strip()
        if not nombre_experimento:
            console.print("El nombre no puede estar vacio.")

    base_salida = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..","..","EEG_controller_app", "recordings", "experimento_RT")
    )

    carpeta_salida = os.path.join(base_salida, nombre_experimento)
    os.makedirs(carpeta_salida, exist_ok=True)

    SalidaPredicciones = os.path.join(carpeta_salida, "SalidaPredicciones.csv")

    console.print(f"Guardando ficheros en: {carpeta_salida}")

    experimento_offline_RT(
        emulation_fif,
        fif_train,
        SalidaPredicciones,
        console,
    )


if __name__ == "__main__":
        
        console = Console()

        fif_train = "EEG_controller_app/recordings/suj2_1_raw.fif"

        emulation_fif = "EEG_controller_app/recordings/suj2_2_raw.fif"

        interfaz_experimento_RT(emulation_fif, fif_train, console)

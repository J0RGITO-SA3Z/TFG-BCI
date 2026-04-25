import mne
import os, sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from components.Trainings.trainingOffline import Training_offline

from components.RT_pipe_components.modules.RT_pipeline_process import RT_pipeline_process
from components.RT_pipe_components.EEGSimulator import EEGSimulator
from components.RT_pipe_components.Interpreters.RT_interpreter_process import RT_interpreter_process

# Interpreters del juego
from components.RT_pipe_components.Interpreters.RT_interpreter_slider import RT_interpreter_slider
from components.RT_pipe_components.Interpreters.RT_interpreter_resampler  import RT_interpreter_resampler

from components.RT_pipe_components.MIGames.MI_game_process import MIGameProcess
from components.RT_pipe_components.MIGames.MIGame import MIGame
from components.RT_pipe_components.MIGames.flappy_bird_mi_game import FlappyBirdMIGame
from components.RT_pipe_components.MIGames.arrow_runner_mi_game import ArrowRunnerMIGame
from components.RT_pipe_components.MIGames.dualArrowComponent import DualArrowComponent


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

    # Primero el juego, para obtener su pipe de entrada antes de crear el interpreter.
    migame_process = MIGameProcess(ArrowRunnerMIGame)
    migame_process.start()

    interpreter_process = RT_interpreter_process(RT_interpreter_slider, game_pipe=migame_process.get_send_pipe())
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
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "recordings", "simulations_RT")
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

        fif_train = "EEG_app/recordings/piloto/suj2/suj2_5_raw.fif"

        emulation_fif = "EEG_app/recordings/piloto/suj2/suj2_6_raw.fif"

        interfaz_experimento_RT(emulation_fif, fif_train, console)

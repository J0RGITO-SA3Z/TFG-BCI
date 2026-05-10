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
from components.RT_pipe_components.MIGames.endless_wave_runner import EndlessWaveRunnerMIGame
from components.RT_pipe_components.MIGames.project_arrows import ProjectArrowsMIGame

import time
from typing import Type
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

GAMES: dict[str, Type[MIGame]] = {
    "1": ProjectArrowsMIGame,
    "2": FlappyBirdMIGame,
    "3": ArrowRunnerMIGame,
    "4": EndlessWaveRunnerMIGame,
    "5": DualArrowComponent,
}

def experimento_offline_RT(emulationfif,Trainingfif, SalidaPredicciones, game_cls: Type[MIGame], console):


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
    #DualArrowComponent
    #ArrowRunnerMIGame
    #FlappyBirdMIGame
    #EndlessWaveRunnerMIGame
    #ProjectArrowsMIGame
    migame_process = MIGameProcess(game_cls)
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
    migame_process.stop()

    print("grabacion terminada")


def interfaz_experimento_RT(emulation_fif, fif_train, console):
    console.clear()
    console.print(
        Panel(
            "[bold]Experimento de Simulacion en Tiempo Real[/bold]\n"
            "Configure el experimento antes de iniciar el flujo RT.",
            border_style="white",
            padding=(1, 2),
        )
    )

    nombre_experimento = ""
    while not nombre_experimento:
        nombre_experimento = Prompt.ask("Nombre del experimento [dim](se creara una carpeta en recordings)[/dim]").strip()
        if not nombre_experimento:
            console.print("[red]El nombre no puede estar vacio.[/red]")

    base_salida = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "recordings", "simulations_RT")
    )

    carpeta_salida = os.path.join(base_salida, nombre_experimento)
    os.makedirs(carpeta_salida, exist_ok=True)

    SalidaPredicciones = os.path.join(carpeta_salida, "SalidaPredicciones.csv")
    console.print(f"[dim]Guardando ficheros en: {carpeta_salida}[/dim]")

    game_cls = choose_game(console)

    experimento_offline_RT(
        emulation_fif,
        fif_train,
        SalidaPredicciones,
        game_cls,
        console,
    )



def choose_game(console=None) -> Type[MIGame]:
    table = Table(title="Selecciona un juego MI", expand=True)
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Juego")

    for key, game_cls in GAMES.items():
        table.add_row(key, game_cls.__name__)

    if console is not None:
        console.print(table)
    else:
        for key, game_cls in GAMES.items():
            print(f"{key}. {game_cls.__name__}")

    while True:
        option = Prompt.ask("Opcion").strip() if console is not None else input("Opción: ").strip()
        game_cls = GAMES.get(option)
        if game_cls is not None:
            return game_cls
        msg = "Opcion no valida. Prueba otra vez."
        if console is not None:
            console.print(f"[red]{msg}[/red]")
        else:
            print(msg)

if __name__ == "__main__":
        
        console = Console()

        fif_train = "EEG_app/recordings/piloto/suj2/suj2_5_raw.fif"

        emulation_fif = "EEG_app/recordings/piloto/suj2/suj2_6_raw.fif"

        interfaz_experimento_RT(emulation_fif, fif_train, console)

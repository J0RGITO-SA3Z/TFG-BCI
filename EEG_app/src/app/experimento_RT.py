import os, sys
from typing import Type
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from components.Trainings.trainingRealTime import Training_real_time
from app.app_utils import seleccionarPuertoCOM
from app.EEGRecorder import EEGRecorder
from brainaccess.core.eeg_manager import EEGManager
from app.tcp.eeg_live_server import EEGLiveServer

# Interpreters del juego
from components.RT_pipe_components.Interpreters.RT_interpreter_slider import RT_interpreter_slider
from components.RT_pipe_components.Interpreters.RT_interpreter_resampler  import RT_interpreter_resampler

# Pipeline y juegos
from components.RT_pipe_components.modules.RT_pipeline_process import RT_pipeline_process
from components.RT_pipe_components.Interpreters.RT_interpreter_process import RT_interpreter_process
from components.RT_pipe_components.MIGames.MI_game_process import MIGameProcess
from components.RT_pipe_components.MIGames.flappy_bird_mi_game import FlappyBirdMIGame
from components.RT_pipe_components.MIGames.arrow_runner_mi_game import ArrowRunnerMIGame
from components.RT_pipe_components.MIGames.dualArrowComponent import DualArrowComponent
from components.RT_pipe_components.MIGames.endless_wave_runner import EndlessWaveRunnerMIGame
from components.RT_pipe_components.MIGames.project_arrows import ProjectArrowsMIGame
from components.RT_pipe_components.MIGames.MIGame import MIGame

import time
from typing import Type
from rich.console import Console

GAMES: dict[str, Type[MIGame]] = {
    "1": ProjectArrowsMIGame,
    "2": FlappyBirdMIGame,
    "3": ArrowRunnerMIGame,
    "4": EndlessWaveRunnerMIGame,
    "5": DualArrowComponent,
}

def choose_game() -> Type[MIGame]:
    print("\n=== Selecciona un juego MI ===")

    for key, game_cls in GAMES.items():
        print(f"{key}. {game_cls.__name__}")

    while True:
        option = input("Opción: ").strip()

        game_cls = GAMES.get(option)
        if game_cls is not None:
            return game_cls

        print("Opción no válida. Prueba otra vez.")

def experimento_RT(channelsConfig, puertoCom, SalidaEntrenamiento, SalidaGeneral, SalidaPredicciones, game_cls: Type[MIGame], console):
    rtTraining = Training_real_time()
    EA_matrix, model = rtTraining.start(
        puerto_COM=puertoCom,
        channelsConfig=channelsConfig,
        lista=["left_hand", "right_hand"],
        fif_name=SalidaEntrenamiento,
        numTrialsClase=30,
        use_bad_channel_interpolator=True,
        min_epochs_per_class=20,
    )

    eeg = EEGRecorder()
    live_server = None
    raw = None
    interpreter_process = None
    modelPipeline = None

    with EEGManager() as mgr:
        eeg.configAndConect(
            mgr=mgr,
            COM_port=puertoCom,
            channelConfig=channelsConfig,
            gain=8,
        )

        console.print("Configurando stream EEG...")
        eeg.iniciarGrabacion()

        live_server = EEGLiveServer(
            ch_names=eeg.get_ch_names_ordered(),
            sfreq=eeg.get_sfreq(),
            ch_types=eeg.get_ch_types_ordered(),
            total_epochs=0,
            initial_action="Empezando",
        )

        # Primero el juego, para obtener su pipe de entrada antes de crear el interpreter.
        #DualArrowComponent 
        #ArrowRunnerMIGame
        #FlappyBirdMIGame
        #EndlessWaveRunnerMIGame
        #ProjectArrowsMIGame
        migame_process = MIGameProcess(game_cls)
        migame_process.start()
        
        # Crea el interprete en proceso separado para recibir predicciones.
        interpreter_process = RT_interpreter_process(RT_interpreter_slider, game_pipe=migame_process.get_send_pipe())
        interpreter_process.start(filename=SalidaPredicciones)

        # Crea el pipeline en proceso separado y conecta su salida al interprete.
        modelPipeline = RT_pipeline_process(
            ea_matrix = EA_matrix,
            pretrained_model = model,
            info = eeg.get_info(),
            channel_positions = eeg.get_channel_indexes(),
        )
        modelPipeline.run_process(interpreter_process)

        live_server.start()
        eeg.register_callback(live_server.newChunk)
        eeg.register_callback(modelPipeline.sendData)
        console.print("Servidor TCP en linea. Esperando conexiones...")

        console.print("Cargando buffer inicial durante 10 segundos antes de activar predicciones...")
        time.sleep(15)
        modelPipeline.activar_predecir()
        console.print("Predicciones activadas.")

        # Menu de control en consola.
        running = True
        while running:
            console.print("\nMenu: [1] Pausar prediccion  [2] Reanudar prediccion  [0] Detener grabacion")
            opcion = input("Selecciona opcion: ").strip()

            if opcion == "1":
                modelPipeline.desactivar_predecir()
                console.print("Prediccion pausada. El buffer sigue recibiendo datos.")
            elif opcion == "2":
                modelPipeline.activar_predecir()
                console.print("Prediccion reanudada.")
            elif opcion == "0":
                console.print("Deteniendo grabacion y procesos...")
                running = False
            else:
                console.print("Opcion no valida.")

        # acaba la grabacion
        raw = eeg.get_mne()
        raw.save(SalidaGeneral, overwrite=True)

        eeg.detenerGrabacion()

        if modelPipeline is not None:
            modelPipeline.stop_process()

        if interpreter_process is not None:
            interpreter_process.stop()

        mgr.disconnect()
        live_server.stop()

    eeg.cerrarLibreria()

def interfaz_experimento_RT(channelsConfig, console):
    console.print("Bienvenido al experimento de entrenamiento y prediccion en tiempo real.")
    console.print("Selecciona el puerto COM al que esta conectado el dispositivo EEG:")
    puertoCom = seleccionarPuertoCOM(console)

    if puertoCom is None:
        console.print("No se selecciono un puerto COM. Cancelando experimento.")
        return

    nombre_experimento = ""
    while not nombre_experimento:
        nombre_experimento = input("Nombre del experimento (se creara una carpeta en recordings): ").strip()
        if not nombre_experimento:
            console.print("El nombre no puede estar vacio.")

    base_salida = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "recordings", "experimento_RT")
    )

    carpeta_salida = os.path.join(base_salida, nombre_experimento)
    os.makedirs(carpeta_salida, exist_ok=True)

    SalidaEntrenamiento = os.path.join(carpeta_salida, "SalidaEntrenamiento.fif")
    SalidaGeneral = os.path.join(carpeta_salida, "SalidaGeneral.fif")
    SalidaPredicciones = os.path.join(carpeta_salida, "SalidaPredicciones.csv")

    console.print(f"Guardando ficheros en: {carpeta_salida}")

    game_cls = choose_game()

    experimento_RT(
        channelsConfig,
        puertoCom,
        SalidaEntrenamiento,
        SalidaGeneral,
        SalidaPredicciones,
        game_cls,
        console,
    )
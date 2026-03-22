from pipeline_RT.trainingRealTime import Training_real_time
from app_utils import seleccionarPuertoCOM
import EEGRecorder
from brainaccess.core.eeg_manager import EEGManager
from tcp.eeg_live_server import EEGLiveServer

from pipeline_RT.RT_modules.RT_pipeline_process import RT_pipeline_process
from pipeline_RT.RT_interpreters.RT_interpreter_process import RT_interpreter_process
import time

def experimento_RT(channelsConfig, puertoCom, SalidaEntrenamiento, SalidaGeneral, console):
    rtTraining = Training_real_time()
    model, EA_matrix = rtTraining.start(puerto_COM=puertoCom, channelsConfig=channelsConfig, fif_name=SalidaEntrenamiento)

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

        # Crea el interprete en proceso separado para recibir predicciones.
        interpreter_process = RT_interpreter_process()
        interpreter_process.start(filename="predictions_log.csv")

        # Crea el pipeline en proceso separado y conecta su salida al interprete.
        modelPipeline = RT_pipeline_process(
            EA_matrix,
            model,
            eeg.get_info(),
            eeg.get_channel_indexes(),
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
from brainaccess.utils.acquisition import EEGData
from brainaccess.core.eeg_manager import EEGManager
from configuracion_canales import ChannelConfig
import threading
import asyncio
import numpy as np

import brainaccess.core as bacore
from brainaccess.core.gain_mode import GainMode, multiplier_to_gain_mode
import brainaccess.core.eeg_channel as eeg_channel
import mne

class EEGRecorder:
    def __init__(self):
        self.channelConfig: list[ChannelConfig] = []
        self.mgr: EEGManager = None
        self.callbackFunctions = []
        self.gain: GainMode = GainMode.X8
        self.lock = threading.Lock()
        self.indices_canales_habilitados = {}
        self.indiceCanales = {}
        self.nombreCanales = {}

        bacore.init(bacore.Version(2, 0, 0))

    def _callbackFunction(self, chunk, chunk_size):
        self.data.data.append(np.array(chunk))

        for func in self.callbackFunctions:
            func(chunk, chunk_size)

    def register_callback(self, func):
        self.callbackFunctions.append(func)
    
    def unregister_callback(self, func):
        self.callbackFunctions.remove(func)
    
    def _init_channel_mappings(self):
        # Inicializa nombreCanales con los nombres de los canales habilitados
        for ch in self.channelConfig:
            if ch.enabled:
                self.nombreCanales[eeg_channel.ELECTRODE_MEASUREMENT + ch.index] = ch.nombre

        self.nombreCanales[eeg_channel.ACCELEROMETER + 0] = "Accel_x"
        self.nombreCanales[eeg_channel.ACCELEROMETER + 1] = "Accel_y"
        self.nombreCanales[eeg_channel.ACCELEROMETER + 2] = "Accel_z"
        self.nombreCanales[eeg_channel.DIGITAL_INPUT] = "Digital"
        self.nombreCanales[eeg_channel.SAMPLE_NUMBER] = "Sample"

        # Inicializa indiceCanales con el numero correspondiente a cada canal en BrainAccess
        for ch in self.channelConfig:
            if ch.enabled:
                self.indiceCanales[eeg_channel.ELECTRODE_MEASUREMENT + ch.index] = 0

        self.indiceCanales[eeg_channel.ACCELEROMETER + 0] = 0
        self.indiceCanales[eeg_channel.ACCELEROMETER + 1] = 0
        self.indiceCanales[eeg_channel.ACCELEROMETER + 2] = 0
        self.indiceCanales[eeg_channel.DIGITAL_INPUT] = 0
        self.indiceCanales[eeg_channel.SAMPLE_NUMBER] = 0

    def _create_info(self):
        sfreq = self.mgr.get_sample_frequency()
        ch_names = [x.electrode for x in self.nombreCanales.values()]

        ch_types = ['eeg'] * len(self.channelConfig)
        ch_types.extend(["misc"] * 3)
        ch_types.append("stim")
        ch_types.append("syst")

        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        info.set_montage("standard_1005")

        return info
    
    def obtenerIndicesCanalesActivos(self):
        for key in self.indiceCanales.keys():
            self.indiceCanales[key] = self.mgr.get_channel_index(key)

    async def configurarEEG(self):
        for ch in self.channelConfig:
            self.mgr.set_channel_enabled(eeg_channel.ELECTRODE_MEASUREMENT + ch.index, ch.enabled)
            self.mgr.set_channel_gain(eeg_channel.ELECTRODE_MEASUREMENT + ch.index, self.gain)

            if ch.is_bias:
                self.mgr.set_channel_bias(eeg_channel.ELECTRODE_MEASUREMENT + ch.index, True)
        
        self.mgr.set_callback_chunk(self._callbackFunction)

    async def _inicarGrabacion(self):
        await self.configurarEEG()

        try:
            await self.mgr.start_stream()
        except Exception:
            raise Exception
        
        self.obtenerIndicesCanalesActivos()

    def iniciarGrabacion(self, mgr:EEGManager, COM_port, channelConfig: list[ChannelConfig], gain):
        asyncio.run(mgr.connect(COM_port))

        self.mgr = mgr
        self.channelConfig = channelConfig
        self.gain = multiplier_to_gain_mode(gain)
        self._init_channel_mappings() # inicializa indiceCanal y nombreCanal
        self.info = self._create_info()
        self.data = EEGData(self.info, lock=self.lock, zeros_at_start=0)

        asyncio.run(self._inicarGrabacion())

    def detenerGrabacion(self):
        asyncio.run(self.mgr.stop_stream())

    def anotar(self, anotacion:str):
        self.mgr.annotate(anotacion)

    def get_anotaciones(self):
        return self.mgr.get_annotations()
    
    def desconectar(self):
        self.mgr.disconnect()

    def cerrarLibreria(self):
        bacore.close()
    
    def get_mne(self, tim = None, samples = None) -> mne.io.BaseRaw:
        self.data.annotations = self.get_anotaciones() # sincroniza anotaciones con los datos

        self.data.convert_to_mne(
            tim=tim,
            samples=samples,
            channels_indexes=list(self.indiceCanales.values()),
        )

        return self.data.mne_raw
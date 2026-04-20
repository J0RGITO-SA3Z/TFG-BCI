import os, sys
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import time
from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager
from app.configuracion_canales import ChannelConfig

MEASUREMENT_SECONDS = 20


class ImpedanceMeter:
    """Encapsula la medición de impedancias con el dispositivo BrainAccess."""

    def __init__(self, puerto_com: str, canales: list[ChannelConfig]):
        self.puerto_com = puerto_com
        self.canales = canales
        self._last_raw: list[float] | None = None

    def measure(self, console=None) -> list[float]:
        """Realiza la medición y devuelve la lista de valores crudos (un valor por canal activo)."""
        electrodes = {ch.index: ch.electrode for ch in self.canales if ch.enabled}
        eeg = acquisition.EEG()

        with EEGManager() as mgr:
            eeg.setup(mgr=mgr, port=self.puerto_com, cap=electrodes, gain=8, bias=[])
            eeg.start_impedance_measurement()

            start = time.time()
            imp = None
            while time.time() - start < MEASUREMENT_SECONDS:
                time.sleep(1)
                imp = eeg.calc_impedances()
                if console:
                    console.print(imp)

            eeg.stop_impedance_measurement()
            mgr.disconnect()

        eeg.close()
        self._last_raw = imp
        return imp

    def to_map(self, raw: list[float] | None = None) -> dict[str, float]:
        """Convierte la lista cruda en {nombre_electrodo: valor}. Usa la última medición si no se pasa raw."""
        source = raw if raw is not None else self._last_raw
        if source is None:
            raise ValueError("No hay medición disponible. Llama a measure() primero.")

        result = {}
        idx = 0
        for ch in self.canales:
            if ch.enabled:
                result[ch.electrode] = source[idx]
                idx += 1
        return result

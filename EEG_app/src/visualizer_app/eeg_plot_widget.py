import threading
import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt
from PyQt5 import QtCore
import pyqtgraph as pg

from brainaccess.utils.acquisition import EEGData_roll

_COLOR_OK_DEFAULT = None          # se sustituye por el color del canal
_COLOR_FLAT = (120, 120, 120)     # gris
_COLOR_NOISY = (255, 80, 80)      # rojo


class EEGPlotWidget(pg.PlotWidget):
    """
    Widget de PyQtGraph que visualiza los datos contenidos en un
    ``EEGData_roll`` de brainaccess.

    Se construye únicamente a partir de un ``mne.Info`` (de donde extrae
    ``ch_names`` y ``sfreq``) y los segundos de buffer deseados.
    Internamente crea un ``EEGData_roll`` con el tamaño adecuado.

    Detección de canal desconectado
    --------------------------------
    - **Flatline** (gris): varianza del canal < ``flat_var_thresh`` µV²
      → cable en cortocircuito o electrodo sin contacto.
    - **Saturado** (rojo): pico a pico > ``noisy_ptp_thresh`` µV
      → electrodo suelto captando interferencia de red/movimiento.
    """

    def __init__(
        self,
        info: mne.Info,
        buffer_seconds: float = 10.0,
        flat_var_thresh: float = 1.0,
        noisy_ptp_thresh: float = 500.0,
        parent=None,
    ):
        super().__init__(parent=parent)

        self.info = info
        self.sfreq: float = info["sfreq"]
        self.channel_names: list[str] = list(info.ch_names)
        self.n_channels: int = len(self.channel_names)
        self.buffer_len: int = int(buffer_seconds * self.sfreq)
        self._buffer_seconds = buffer_seconds

        self._flat_var_thresh = flat_var_thresh
        self._noisy_ptp_thresh = noisy_ptp_thresh

        self.lock = threading.Lock()
        self.eeg_data = EEGData_roll(
            info=info,
            lock=self.lock,
            zeros_at_start=self.buffer_len,
        )

        self._sos_filter = butter(4, [1.0, 40.0], btype='band', fs=self.sfreq, output='sos')

        cmap = pg.colormap.get("CET-C6")
        self.colors = [
            cmap.map(i / max(self.n_channels - 1, 1), mode="qcolor")
            for i in range(self.n_channels)
        ]

        self.setBackground("k")
        self.showGrid(x=True, y=False, alpha=0.15)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setLabel("bottom", "Tiempo", units="s")

        y_axis = self.getAxis("left")
        y_axis.setTicks([[(i, self.channel_names[i]) for i in range(self.n_channels)]])
        y_axis.setStyle(tickLength=0)

        for i in range(self.n_channels):
            guide = pg.InfiniteLine(
                pos=i,
                angle=0,
                pen=pg.mkPen(color=(180, 180, 180, 70), width=1, style=QtCore.Qt.DashLine),
            )
            self.addItem(guide)
            self._guides.append(guide)

        self._guides: list[pg.InfiniteLine] = []
        self.curves: list[pg.PlotDataItem] = []
        for i in range(self.n_channels):
            curve = self.plot(pen=pg.mkPen(self.colors[i], width=1.2))
            self.curves.append(curve)

        self.setXRange(0, buffer_seconds, padding=0)
        self.setYRange(-0.5, self.n_channels - 0.5, padding=0.02)

        self._add_legend()

        # Caché del último estado por canal para no repintar el pen en cada tick
        self._last_status: list[str] = ["ok"] * self.n_channels

    # ------------------------------------------------------------------ #
    def _add_legend(self):
        """Añade una leyenda de estado de canal en la esquina superior derecha."""
        html = (
            '<div style="text-align:right; line-height:1.6;">'
            '<span style="color:#7ecbff;">━</span>'
            '<span style="color:#aaa; font-size:10px;"> Normal &nbsp;&nbsp;</span>'
            '<span style="color:#787878;">━</span>'
            '<span style="color:#aaa; font-size:10px;"> Flatline &nbsp;&nbsp;</span>'
            '<span style="color:#ff5050;">━</span>'
            '<span style="color:#aaa; font-size:10px;"> Saturado</span>'
            '</div>'
        )
        self._legend_item = pg.TextItem(html=html, anchor=(1, 0))
        self._legend_item.setPos(self._buffer_seconds, self.n_channels - 0.5)
        self.addItem(self._legend_item)

    # ------------------------------------------------------------------ #
    def _channel_status(self, ch: np.ndarray) -> str:
        """Devuelve 'flat', 'noisy' u 'ok' según las métricas del canal."""
        if np.var(ch) < self._flat_var_thresh:
            return "flat"
        if np.ptp(ch) > self._noisy_ptp_thresh:
            return "noisy"
        return "ok"

    # ------------------------------------------------------------------ #
    def push_chunk(self, chunk: np.ndarray, chunk_size: int):
        """
        Añade un bloque de datos al ``EEGData_roll``.

        Parameters
        ----------
        chunk : np.ndarray  (n_channels, chunk_size)
        chunk_size : int
        """
        with self.lock:
            self.eeg_data.data = np.roll(self.eeg_data.data, -chunk_size, axis=1)
            self.eeg_data.data[:, -chunk_size:] = chunk

    # ------------------------------------------------------------------ #
    def refresh(self):
        """Lee el buffer del EEGData_roll, filtra 1-40 Hz, reescala y repinta."""
        with self.lock:
            data = self.eeg_data.data.copy()

        try:
            data = sosfiltfilt(self._sos_filter, data, axis=1)
        except ValueError:
            pass  # buffer demasiado corto al inicio

        t = np.arange(self.buffer_len) / self.sfreq

        for i in range(self.n_channels):
            ch = data[i]

            status = self._channel_status(ch)
            if status != self._last_status[i]:
                self._last_status[i] = status
                if status == "flat":
                    color = _COLOR_FLAT
                elif status == "noisy":
                    color = _COLOR_NOISY
                else:
                    color = self.colors[i]
                self.curves[i].setPen(pg.mkPen(color, width=1.2))

            ch_min, ch_max = ch.min(), ch.max()
            rng = ch_max - ch_min
            if rng < 1e-12:
                normed = np.zeros_like(ch)
            else:
                normed = (ch - ch_min) / rng * 0.8 - 0.4
            self.curves[i].setData(t, normed + i)

    # ------------------------------------------------------------------ #
    def reset_info(self, info: mne.Info):
        """
        Actualiza el widget con una nueva configuración de canales.
        Resetea el buffer a ceros. Si cambia el número de canales reconstruye
        las curvas; si solo cambian los nombres actualiza las etiquetas.
        """
        new_n = len(info.ch_names)
        with self.lock:
            self.info = info
            self.sfreq = info["sfreq"]
            self.channel_names = list(info.ch_names)

            if new_n != self.n_channels:
                self.n_channels = new_n
                self.buffer_len = int(self._buffer_seconds * self.sfreq)

                self.eeg_data = EEGData_roll(
                    info=info,
                    lock=self.lock,
                    zeros_at_start=self.buffer_len,
                )
                self._sos_filter = butter(4, [1.0, 40.0], btype='band', fs=self.sfreq, output='sos')

                cmap = pg.colormap.get("CET-C6")
                self.colors = [
                    cmap.map(i / max(self.n_channels - 1, 1), mode="qcolor")
                    for i in range(self.n_channels)
                ]

                for curve in self.curves:
                    self.removeItem(curve)
                for guide in self._guides:
                    self.removeItem(guide)

                self._guides = []
                for i in range(self.n_channels):
                    guide = pg.InfiniteLine(
                        pos=i,
                        angle=0,
                        pen=pg.mkPen(color=(180, 180, 180, 70), width=1, style=QtCore.Qt.DashLine),
                    )
                    self.addItem(guide)
                    self._guides.append(guide)

                self.curves = []
                for i in range(self.n_channels):
                    curve = self.plot(pen=pg.mkPen(self.colors[i], width=1.2))
                    self.curves.append(curve)

                self.setYRange(-0.5, self.n_channels - 0.5, padding=0.02)
                self._add_legend()
            else:
                self.eeg_data.data[:] = 0

            y_axis = self.getAxis("left")
            y_axis.setTicks([[(i, self.channel_names[i]) for i in range(self.n_channels)]])
            self._last_status = ["ok"] * self.n_channels
            for i, curve in enumerate(self.curves):
                curve.setPen(pg.mkPen(self.colors[i], width=1.2))

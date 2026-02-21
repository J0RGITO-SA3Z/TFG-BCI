"""
Visualización de canales EEG en tiempo real con PyQtGraph.
- Usa EEGData_roll de brainaccess como almacén de datos
- Recibe un mne.Info para configurarse automáticamente
- Reescalado automático por canal con offsets verticales
- Generador de datos aleatorios para pruebas standalone
"""

import sys
import json
import socket
import threading
import numpy as np
import mne
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from brainaccess.utils.acquisition import EEGData_roll


# ---------------------------------------------------------------------- #
# Generador de datos sintéticos (solo para pruebas)
# ---------------------------------------------------------------------- #
def generate_random_eeg(n_channels: int, n_samples: int, sfreq: float = 250.0) -> np.ndarray:
    """
    Genera datos EEG sintéticos con forma similar a señales reales.

    Returns
    -------
    np.ndarray  (n_channels, n_samples)
    """
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        alpha = np.random.uniform(5, 30) * np.sin(
            2 * np.pi * (9 + np.random.randn()) * t + np.random.uniform(0, 2 * np.pi)
        )
        beta = np.random.uniform(2, 10) * np.sin(
            2 * np.pi * (20 + np.random.randn()) * t + np.random.uniform(0, 2 * np.pi)
        )
        white = np.random.randn(n_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
        freqs[0] = 1
        fft /= np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=n_samples) * 5
        data[ch] = alpha + beta + pink
    return data


# ---------------------------------------------------------------------- #
# Widget de visualización
# ---------------------------------------------------------------------- #
class EEGPlotWidget(pg.PlotWidget):
    """
    Widget de PyQtGraph que visualiza los datos contenidos en un
    ``EEGData_roll`` de brainaccess.

    Se construye únicamente a partir de un ``mne.Info`` (de donde extrae
    ``ch_names`` y ``sfreq``) y los segundos de buffer deseados.
    Internamente crea un ``EEGData_roll`` con el tamaño adecuado.
    """

    def __init__(
        self,
        info: mne.Info,
        buffer_seconds: float = 10.0,
        parent=None,
    ):
        super().__init__(parent=parent)

        # ---- Metadatos desde mne.Info -------------------------------- #
        self.info = info
        self.sfreq: float = info["sfreq"]
        self.channel_names: list[str] = list(info.ch_names)
        self.n_channels: int = len(self.channel_names)
        self.buffer_len: int = int(buffer_seconds * self.sfreq)

        # ---- EEGData_roll como almacén de datos ---------------------- #
        self.lock = threading.Lock()
        self.eeg_data = EEGData_roll(
            info=info,
            lock=self.lock,
            zeros_at_start=self.buffer_len,
        )

        # ---- Apariencia --------------------------------------------- #
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

        # Eje Y con nombres de canales
        y_axis = self.getAxis("left")
        y_axis.setTicks([[(i, self.channel_names[i]) for i in range(self.n_channels)]])
        y_axis.setStyle(tickLength=0)

        # Líneas guía horizontales (una por canal, gris semitransparente)
        for i in range(self.n_channels):
            guide = pg.InfiniteLine(
                pos=i,
                angle=0,
                pen=pg.mkPen(color=(180, 180, 180, 70), width=1, style=QtCore.Qt.DashLine),
            )
            self.addItem(guide)

        # Curvas
        self.curves: list[pg.PlotDataItem] = []
        for i in range(self.n_channels):
            curve = self.plot(pen=pg.mkPen(self.colors[i], width=1.2))
            self.curves.append(curve)

        # Rangos fijos
        self.setXRange(0, buffer_seconds, padding=0)
        self.setYRange(-0.5, self.n_channels - 0.5, padding=0.02)

    # ------------------------------------------------------------------ #
    def push_chunk(self, chunk: np.ndarray, chunk_size: int):
        """
        Añade un bloque de datos al ``EEGData_roll`` usando la misma
        mecánica que ``EEG._acq_roll`` de brainaccess:
        roll del buffer + escritura en las últimas posiciones.

        Parameters
        ----------
        chunk : np.ndarray
            Datos con forma ``(n_channels, chunk_size)``.
        chunk_size : int
            Número de muestras nuevas.
        """
        with self.lock:
            self.eeg_data.data = np.roll(self.eeg_data.data, -chunk_size, axis=1)
            self.eeg_data.data[:, -chunk_size:] = chunk

    # ------------------------------------------------------------------ #
    def refresh(self):
        """Lee el buffer del EEGData_roll, reescala y repinta las curvas."""
        with self.lock:
            data = self.eeg_data.data.copy()

        t = np.arange(self.buffer_len) / self.sfreq

        for i in range(self.n_channels):
            ch = data[i]
            ch_min, ch_max = ch.min(), ch.max()
            rng = ch_max - ch_min
            if rng < 1e-12:
                normed = np.zeros_like(ch)
            else:
                normed = (ch - ch_min) / rng * 0.8 - 0.4
            self.curves[i].setData(t, normed + i)


# ---------------------------------------------------------------------- #
# Ventana principal
# ---------------------------------------------------------------------- #
class EEGWindow(QtWidgets.QMainWindow):
    """
    Ventana que contiene un ``EEGPlotWidget`` y un timer de refresco.

    Parameters
    ----------
    info : mne.Info
        Metadatos MNE (canales, sfreq, montaje…).
    buffer_seconds : float
        Segundos visibles en pantalla.
    update_ms : int
        Intervalo de refresco del gráfico (ms).
    """

    def __init__(
        self,
        info: mne.Info,
        buffer_seconds: float = 10.0,
        update_ms: int = 40,
    ):
        super().__init__()
        self.setWindowTitle("EEG Live Viewer")
        self.resize(1200, 700)

        self.info = info
        self.sfreq: float = info["sfreq"]
        self.n_channels: int = len(info.ch_names)
        self.update_ms = update_ms

        # Widget central
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Plot (internamente crea su propio EEGData_roll)
        self.plot_widget = EEGPlotWidget(
            info=info,
            buffer_seconds=buffer_seconds,
        )
        layout.addWidget(self.plot_widget)

        # ---- Fila inferior: progreso + acción, lado a lado ----------- #
        bottom_row = QtWidgets.QHBoxLayout()

        # -- Panel 1: Avance Grabación -------------------------------- #
        progress_frame = QtWidgets.QFrame()
        progress_frame.setObjectName("progressPanel")
        progress_frame.setStyleSheet(
            """
            #progressPanel {
                border: 1px solid #444;
                border-radius: 8px;
                background-color: #1e1e2e;
            }
            """
        )

        progress_inner = QtWidgets.QHBoxLayout(progress_frame)
        progress_inner.setContentsMargins(10, 6, 10, 6)

        progress_label = QtWidgets.QLabel("Avance Grabación")
        progress_label.setStyleSheet("color: #ccc; font-size: 13px;")
        progress_inner.addWidget(progress_label)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(10)
        self._progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #333;
                border-radius: 5px;
                background-color: #111;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e90ff, stop:1 #00bfff
                );
                border-radius: 4px;
            }
            """
        )
        progress_inner.addWidget(self._progress_bar, stretch=1)

        self._progress_text = QtWidgets.QLabel("0/0")
        self._progress_text.setStyleSheet("color: #ccc; font-size: 13px;")
        self._progress_text.setMinimumWidth(50)
        self._progress_text.setAlignment(QtCore.Qt.AlignCenter)
        progress_inner.addWidget(self._progress_text)

        bottom_row.addWidget(progress_frame, stretch=3)

        # Estado interno de progreso
        self._total_epochs = 0
        self._current_epoch = 0
        self.set_total_epochs(40)
        self.set_current_epoch(17)  #demo

        # -- Panel 2: Acción en curso + cronómetro -------------------- #
        action_frame = QtWidgets.QFrame()
        action_frame.setObjectName("actionPanel")
        action_frame.setStyleSheet(
            """
            #actionPanel {
                border: 1px solid #444;
                border-radius: 8px;
                background-color: #1e1e2e;
            }
            """
        )
        action_inner = QtWidgets.QHBoxLayout(action_frame)
        action_inner.setContentsMargins(10, 6, 10, 6)

        self._action_label = QtWidgets.QLabel("Reposo:")
        self._action_label.setStyleSheet(
            "color: #00bfff; font-size: 13px; font-weight: bold;"
        )
        self._action_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        action_inner.addWidget(self._action_label, stretch=1)

        self._action_timer_label = QtWidgets.QLabel("00:00")
        self._action_timer_label.setStyleSheet(
            "color: #ccc; font-size: 13px; font-family: monospace;"
        )
        self._action_timer_label.setMinimumWidth(50)
        self._action_timer_label.setAlignment(QtCore.Qt.AlignCenter)
        action_inner.addWidget(self._action_timer_label)

        action_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred
        )
        bottom_row.addWidget(action_frame, stretch=0)

        layout.addLayout(bottom_row)

        # Cronómetro interno para la acción
        self._action_elapsed = QtCore.QElapsedTimer()
        self._action_elapsed.start()          # arranca ya para que isValid() sea True
        self._action_clock = QtCore.QTimer()
        self._action_clock.timeout.connect(self._tick_action_timer)
        self._action_clock.start(200)

        # Timer de refresco visual (no de adquisición)
        self._refresh_timer = QtCore.QTimer()
        self._refresh_timer.timeout.connect(self.plot_widget.refresh)

        # Timer opcional para inyección de datos de prueba
        self._data_timer = QtCore.QTimer()
        self._data_timer.timeout.connect(self._inject_random)
        self._data_callback = None

    # ------------------------------------------------------------------ #
    # Progreso de grabación
    # ------------------------------------------------------------------ #
    def set_total_epochs(self, total: int):
        """Define el número total de epochs a grabar."""
        self._total_epochs = total
        self._progress_bar.setMaximum(total)
        self._update_progress_text()

    def set_current_epoch(self, current: int):
        """Actualiza el número de epochs ya grabados."""
        self._current_epoch = min(current, self._total_epochs)
        self._progress_bar.setValue(self._current_epoch)
        self._update_progress_text()

    def _update_progress_text(self):
        self._progress_text.setText(f"{self._current_epoch}/{self._total_epochs}")

    # ------------------------------------------------------------------ #
    # Acción en curso
    # ------------------------------------------------------------------ #
    def set_action(self, text: str):
        """Cambia el texto de la acción en curso y reinicia el cronómetro."""
        self._action_label.setText(text)
        self._action_elapsed.start()

    def _tick_action_timer(self):
        """Actualiza la etiqueta del cronómetro de la acción."""
        if self._action_elapsed.isValid():
            elapsed_ms = self._action_elapsed.elapsed()
            secs = int(elapsed_ms / 1000)
            mins = secs // 60
            secs = secs % 60
            self._action_timer_label.setText(f"{mins:02d}:{secs:02d}")

    # ------------------------------------------------------------------ #
    # Recepción de epoch desde TCP
    # ------------------------------------------------------------------ #
    def _on_epoch_received(self, current: int, total: int):
        """Slot para actualizar progreso desde el hilo TCP."""
        self.set_total_epochs(total)
        self.set_current_epoch(current)

    # ------------------------------------------------------------------ #
    def start(self, demo: bool = True):
        """
        Inicia el refresco del gráfico.

        Parameters
        ----------
        demo : bool
            Si True genera datos aleatorios de prueba internamente.
            Si False solo inicia el refresco visual (los datos llegan
            por TCP u otra vía externa).
        """
        self._refresh_timer.start(self.update_ms)

        if demo:
            self._data_timer.start(self.update_ms)

    def _inject_random(self):
        """Genera e inyecta un bloque de datos aleatorios (modo demo)."""
        if self._data_callback:
            chunk = self._data_callback()
        else:
            n_samples = max(1, int(self.sfreq * self.update_ms / 1000))
            chunk = generate_random_eeg(self.n_channels, n_samples, self.sfreq)
        self.plot_widget.push_chunk(chunk, chunk.shape[1])


# ---------------------------------------------------------------------- #
# Diálogo de conexión
# ---------------------------------------------------------------------- #
class ConnectionDialog(QtWidgets.QDialog):
    """Pide IP, puerto y permite elegir modo demo (sin conexión)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Conectar al servidor EEG")
        self.setMinimumWidth(340)
        self.setStyleSheet(
            """
            QDialog   { background-color: #1e1e2e; color: #ccc; }
            QLabel    { color: #ccc; font-size: 13px; }
            QLineEdit, QSpinBox {
                background-color: #111; color: #eee;
                border: 1px solid #444; border-radius: 4px;
                padding: 4px;
            }
            QCheckBox { color: #ccc; font-size: 13px; }
            QPushButton {
                background-color: #1e90ff; color: white;
                border: none; border-radius: 4px; padding: 6px 18px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #00bfff; }
            """
        )

        layout = QtWidgets.QFormLayout(self)

        self.host_edit = QtWidgets.QLineEdit("127.0.0.1")
        self.port_edit = QtWidgets.QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(12345)

        self.demo_check = QtWidgets.QCheckBox("Modo demo (datos aleatorios, sin conexión)")

        layout.addRow("IP:", self.host_edit)
        layout.addRow("Puerto:", self.port_edit)
        layout.addRow(self.demo_check)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        # Deshabilitar campos de red cuando demo está marcado
        self.demo_check.toggled.connect(lambda on: (
            self.host_edit.setEnabled(not on),
            self.port_edit.setEnabled(not on),
        ))

    def get_connection(self):
        """Devuelve (host, port, demo)."""
        return self.host_edit.text().strip(), self.port_edit.value(), self.demo_check.isChecked()


# ---------------------------------------------------------------------- #
# Hilo receptor TCP
# ---------------------------------------------------------------------- #
class TCPReceiverThread(QtCore.QThread):
    """
    Hilo que lee mensajes JSON delimitados por ``\\n`` desde un socket ya
    conectado y emite señales Qt para cada tipo de mensaje.

    Protocolo (JSON-lines):
    -----------------------
    Cada línea es un objeto JSON con un campo ``"type"``.

    ``init``  (primer mensaje obligatorio — se lee antes de crear este hilo)
        ``{"type":"init", "ch_names":["Fp1","Fp2",...], "sfreq":250,
          "total_epochs":40, "action":"Reposo:"}``

    ``action``
        ``{"type":"action", "text":"Mano izquierda:"}``

    ``epoch``
        ``{"type":"epoch", "current":3, "total":40}``

    ``data``
        ``{"type":"data", "samples":[[ch0_s0,ch0_s1,...],[ch1_s0,...],...]}``
        ``samples`` tiene forma (n_channels, n_samples).
    """

    action_received = QtCore.pyqtSignal(str)
    epoch_received = QtCore.pyqtSignal(int, int)
    data_received = QtCore.pyqtSignal(np.ndarray)
    disconnected = QtCore.pyqtSignal()

    def __init__(self, sock: socket.socket, initial_buffer: str = "", parent=None):
        super().__init__(parent)
        self._sock = sock
        self._buffer = initial_buffer
        self._running = True

    # ------------------------------------------------------------------ #
    def run(self):
        self._sock.settimeout(1.0)

        while self._running:
            # Procesar mensajes completos que ya estén en el buffer
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip()
                if line:
                    try:
                        self._dispatch(json.loads(line))
                    except json.JSONDecodeError:
                        pass

            # Leer más datos del socket
            try:
                raw = self._sock.recv(8192)
                if not raw:
                    break
                self._buffer += raw.decode("utf-8")
            except socket.timeout:
                continue
            except Exception:
                break

        try:
            self._sock.close()
        except Exception:
            pass
        self.disconnected.emit()

    # ------------------------------------------------------------------ #
    def _dispatch(self, msg: dict):
        t = msg.get("type", "")
        if t == "action":
            self.action_received.emit(msg.get("text", ""))
        elif t == "epoch":
            self.epoch_received.emit(
                int(msg.get("current", 0)),
                int(msg.get("total", 0)),
            )
        elif t == "data":
            samples = np.asarray(msg["samples"], dtype=np.float64)
            if samples.ndim == 2:
                self.data_received.emit(samples)

    # ------------------------------------------------------------------ #
    def stop(self):
        self._running = False
        self.wait(3000)


# ---------------------------------------------------------------------- #
# Ejecución standalone / cliente TCP
# ---------------------------------------------------------------------- #
def ejecutar_cliente_visualizacion():
    app = QtWidgets.QApplication(sys.argv)

    # -- Diálogo de conexión ------------------------------------------ #
    dialog = ConnectionDialog()
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        sys.exit(0)

    host, port, demo = dialog.get_connection()

    # ================================================================== #
    #  Modo DEMO (datos aleatorios, sin conexión)
    # ================================================================== #
    if demo:
        channel_names = [
            "Fp1", "Fp2", "F3", "Fz", "F4",
            "C3", "Cz", "C4", "P3", "Pz",
            "P4", "O1", "O2", "T7", "T8",
        ]
        info = mne.create_info(ch_names=channel_names, sfreq=250.0, ch_types="eeg")
        info.set_montage("standard_1005")

        win = EEGWindow(info=info, buffer_seconds=6.0, update_ms=10)
        win.start(demo=True)
        win.show()
        sys.exit(app.exec_())
    else:
        # ================================================================== #
        #  Modo TCP — conectar y esperar Init
        # ================================================================== #
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((host, port))

            # Leer hasta obtener el mensaje Init
            buf = ""
            init_msg = None
            while init_msg is None:
                raw = sock.recv(4096)
                if not raw:
                    raise ConnectionError("Conexión cerrada antes de recibir Init")
                buf += raw.decode("utf-8")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    msg = json.loads(line)
                    if msg.get("type") == "init":
                        init_msg = msg
                        break
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error de conexión", str(e))
            sys.exit(1)

        # Construir mne.Info desde Init
        ch_names = init_msg["ch_names"]
        ch_types = init_msg.get("ch_types", "eeg")
        sfreq = float(init_msg["sfreq"])
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        info.set_montage(init_msg.get("montage", "standard_1005"))

        # Crear ventana
        win = EEGWindow(
            info=info,
            buffer_seconds=init_msg.get("buffer_seconds", 6.0),
            update_ms=10,
        )
        win.set_total_epochs(init_msg.get("total_epochs", 0))
        win.set_current_epoch(0)
        if init_msg.get("action"):
            win.set_action(init_msg["action"])

        # Hilo receptor TCP (le pasamos el socket ya conectado + buffer sobrante)
        receiver = TCPReceiverThread(sock, buf)
        receiver.action_received.connect(win.set_action)
        receiver.epoch_received.connect(win._on_epoch_received)
        receiver.data_received.connect(
            lambda d: win.plot_widget.push_chunk(d, d.shape[1])
        )
        receiver.disconnected.connect(win.close)
        receiver.start()

        win.start(demo=False)  # solo refresco visual, datos vienen por TCP
        win.show()

        ret = app.exec_()
        receiver.stop()
        sys.exit(ret)


if __name__ == "__main__":
    ejecutar_cliente_visualizacion()

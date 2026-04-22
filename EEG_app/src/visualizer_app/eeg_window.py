import mne
from PyQt5 import QtWidgets, QtCore

from .eeg_plot_widget import EEGPlotWidget


class EEGWindow(QtWidgets.QMainWindow):
    """
    Ventana que contiene un ``EEGPlotWidget`` y un timer de refresco.

    Parameters
    ----------
    info : mne.Info
    buffer_seconds : float
    update_ms : int
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

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.plot_widget = EEGPlotWidget(info=info, buffer_seconds=buffer_seconds)
        layout.addWidget(self.plot_widget)

        # ---- Fila inferior: progreso + acción ------------------------ #
        bottom_row = QtWidgets.QHBoxLayout()

        # -- Panel progreso grabación ---------------------------------- #
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

        self._total_epochs = 0
        self._current_epoch = 0

        # -- Panel acción en curso + cronómetro ----------------------- #
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

        self._action_elapsed = QtCore.QElapsedTimer()
        self._action_elapsed.start()
        self._action_clock = QtCore.QTimer()
        self._action_clock.timeout.connect(self._tick_action_timer)
        self._action_clock.start(200)

        self._refresh_timer = QtCore.QTimer()
        self._refresh_timer.timeout.connect(self.plot_widget.refresh)

    # ------------------------------------------------------------------ #
    # Progreso de grabación
    # ------------------------------------------------------------------ #
    def set_total_epochs(self, total: int):
        self._total_epochs = total
        self._progress_bar.setMaximum(total)
        self._update_progress_text()

    def set_current_epoch(self, current: int):
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
        if self._action_elapsed.isValid():
            elapsed_ms = self._action_elapsed.elapsed()
            secs = int(elapsed_ms / 1000)
            mins = secs // 60
            secs = secs % 60
            self._action_timer_label.setText(f"{mins:02d}:{secs:02d}")

    # ------------------------------------------------------------------ #
    # Slot para actualizar progreso desde el hilo TCP
    # ------------------------------------------------------------------ #
    def _on_epoch_received(self, current: int, total: int):
        self.set_total_epochs(total)
        self.set_current_epoch(current)

    # ------------------------------------------------------------------ #
    def _on_info_received(self, msg: dict):
        """Actualiza la configuración de canales a mitad de grabación."""
        ch_names = msg.get("ch_names", list(self.info.ch_names))
        ch_types = msg.get("ch_types", ["misc"] * len(ch_names))
        sfreq = float(msg.get("sfreq", self.sfreq))
        new_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        self.info = new_info
        self.sfreq = sfreq
        self.n_channels = len(ch_names)
        self.plot_widget.reset_info(new_info)

    # ------------------------------------------------------------------ #
    def start(self):
        """Inicia el refresco visual del gráfico."""
        self._refresh_timer.start(self.update_ms)

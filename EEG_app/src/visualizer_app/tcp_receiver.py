import json
import socket
import numpy as np
from PyQt5 import QtCore


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
    info_received = QtCore.pyqtSignal(dict)
    disconnected = QtCore.pyqtSignal()

    def __init__(self, sock: socket.socket, initial_buffer: str = "", parent=None):
        super().__init__(parent)
        self._sock = sock
        self._buffer = initial_buffer
        self._running = True

    def run(self):
        self._sock.settimeout(1.0)

        while self._running:
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip()
                if line:
                    try:
                        self._dispatch(json.loads(line))
                    except json.JSONDecodeError:
                        pass

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

    def _dispatch(self, msg: dict):
        t = msg.get("type", "")
        if t == "action":
            self.action_received.emit(msg.get("text", ""))
        elif t == "epoch":
            self.epoch_received.emit(
                int(msg.get("current", 0)),
                int(msg.get("total", 0)),
            )
        elif t == "info":
            self.info_received.emit({k: v for k, v in msg.items() if k != "type"})
        elif t == "data":
            samples = np.asarray(msg["samples"], dtype=np.float64)
            if samples.ndim == 2:
                self.data_received.emit(samples)

    def stop(self):
        self._running = False
        self.wait(3000)
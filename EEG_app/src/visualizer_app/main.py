import sys
import json
import socket
import mne
from PyQt5 import QtWidgets

from .connection_dialog import ConnectionDialog
from .eeg_window import EEGWindow
from .tcp_receiver import TCPReceiverThread


def ejecutar_cliente_visualizacion():
    app = QtWidgets.QApplication(sys.argv)

    dialog = ConnectionDialog()
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        sys.exit(0)

    host, port = dialog.get_connection()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect((host, port))

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

    ch_names = init_msg["ch_names"]
    ch_types = init_msg.get("ch_types", "eeg")
    sfreq = float(init_msg["sfreq"])
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage(init_msg.get("montage", "standard_1005"))

    win = EEGWindow(
        info=info,
        buffer_seconds=init_msg.get("buffer_seconds", 6.0),
        update_ms=10,
    )
    win.set_total_epochs(init_msg.get("total_epochs", 0))
    win.set_current_epoch(0)
    if init_msg.get("action"):
        win.set_action(init_msg["action"])

    receiver = TCPReceiverThread(sock, buf)
    receiver.action_received.connect(win.set_action)
    receiver.epoch_received.connect(win._on_epoch_received)
    receiver.info_received.connect(win._on_info_received)
    receiver.data_received.connect(
        lambda d: win.plot_widget.push_chunk(d, d.shape[1])
    )
    receiver.disconnected.connect(win.close)
    receiver.start()

    win.start()
    win.show()

    ret = app.exec_()
    receiver.stop()
    sys.exit(ret)


if __name__ == "__main__":
    ejecutar_cliente_visualizacion()

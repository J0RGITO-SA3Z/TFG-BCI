import json
from pathlib import Path
from PyQt5 import QtWidgets

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default" / "default_tcp.json"
_DEFAULTS = {"host": "127.0.0.1", "port": 12345}


def _load_tcp_config() -> dict:
    try:
        return {**_DEFAULTS, **json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))}
    except Exception:
        return dict(_DEFAULTS)


def _save_tcp_config(host: str, port: int) -> None:
    try:
        _CONFIG_PATH.write_text(
            json.dumps({"host": host, "port": port}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


class ConnectionDialog(QtWidgets.QDialog):
    """Pide IP y puerto para conectar al servidor EEG."""

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
            QPushButton {
                background-color: #1e90ff; color: white;
                border: none; border-radius: 4px; padding: 6px 18px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #00bfff; }
            """
        )

        saved = _load_tcp_config()

        layout = QtWidgets.QFormLayout(self)

        self.host_edit = QtWidgets.QLineEdit(saved["host"])
        self.port_edit = QtWidgets.QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(saved["port"])

        layout.addRow("IP:", self.host_edit)
        layout.addRow("Puerto:", self.port_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def accept(self):
        _save_tcp_config(self.host_edit.text().strip(), self.port_edit.value())
        super().accept()

    def get_connection(self) -> tuple[str, int]:
        """Devuelve (host, port)."""
        return self.host_edit.text().strip(), self.port_edit.value()

"""
Servidor TCP para streaming de datos EEG hacia eeg_live_pyqtgraph.

Uso
---
Desde otro proceso (p.ej. tu pipeline de adquisición)::

    from multiprocessing import Process, Queue
    from eeg_live_server import run_server

    data_queue = Queue()        # (np.ndarray) chunks EEG  (n_ch, n_samples)
    control_queue = Queue()     # dicts: {"action": "..."} ó {"epoch": current, "total": total}

    p = Process(
        target=run_server,
        args=(
            ["Fp1","Fp2","F3","Fz","F4","C3","Cz","C4",
             "P3","Pz","P4","O1","O2","T7","T8"],
            250.0,          # sfreq
            40,             # total_epochs
            "Reposo:",      # acción inicial
            data_queue,
            control_queue,
        ),
    )
    p.start()

    # Enviar datos
    data_queue.put(chunk_ndarray)               # (n_ch, n_samples)
    control_queue.put({"action": "Mano izq:"})
    control_queue.put({"epoch": 5, "total": 40})

Protocolo  (JSON-lines, delimitado por \\n)
-------------------------------------------
init   → {"type":"init","ch_names":[...],"sfreq":250,"total_epochs":40,"action":"Reposo:"}
action → {"type":"action","text":"..."}
epoch  → {"type":"epoch","current":N,"total":M}
data   → {"type":"data","samples":[[...], ...]}
"""

from __future__ import annotations

import json
import socket
import logging
import threading
import numpy as np
from multiprocessing import Process, Queue
from typing import Sequence

import time
from multiprocessing import Process, Queue as MPQueue

PORT = 12345
_SENTINEL = None  # valor que se pone en las colas para pedir parada
log = logging.getLogger("eeg_live_server")

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def _send_line(sock: socket.socket, obj: dict) -> bool:
    """Serializa *obj* como JSON + '\\n' y lo envía.  Devuelve False si falla."""
    try:
        sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
        return True
    except Exception:
        return False

# ------------------------------------------------------------------ #
#  Hilo: atiende a UN cliente conectado
# ------------------------------------------------------------------ #
class _ClientHandler(threading.Thread):
    """Mantiene la conexión con un único cliente y se marca como activo."""

    def __init__(self, conn: socket.socket, addr, init_msg: dict):
        super().__init__(daemon=True)
        self.conn = conn
        self.addr = addr
        self._init_msg = init_msg
        self.alive = True

    def run(self):
        log.info("Cliente conectado desde %s", self.addr)
        if not _send_line(self.conn, self._init_msg):
            self.alive = False
            return

        # Mantener vivo: si el cliente cierra, recv devolverá b""
        self.conn.settimeout(1.0)
        while self.alive:
            try:
                data = self.conn.recv(64)
                if not data:
                    break
            except socket.timeout:
                continue
            except Exception:
                break

        self.alive = False
        try:
            self.conn.close()
        except Exception:
            pass
        log.info("Cliente desconectado: %s", self.addr)

    def send(self, obj: dict) -> bool:
        if not self.alive:
            return False
        ok = _send_line(self.conn, obj)
        if not ok:
            self.alive = False
        return ok


# ------------------------------------------------------------------ #
#  Bucle principal del servidor (se ejecuta en un Process)
# ------------------------------------------------------------------ #
def _server_loop(
    ch_names: list[str],
    ch_types: str | list[str],
    sfreq: float,
    total_epochs: int,
    initial_action: str,
    data_queue: Queue,
    control_queue: Queue,
):
    logging.basicConfig(
        level=logging.INFO,
        format="[EEG-server] %(levelname)s  %(message)s",
    )

    init_msg: dict = {
        "type": "init",
        "ch_names": ch_names,
        "ch_types": ch_types,
        "sfreq": sfreq,
        "total_epochs": total_epochs,
        "action": initial_action,
    }

    # ---- Socket de escucha --------------------------------------- #
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(1)
    srv.settimeout(0.5)
    log.info("Escuchando en 0.0.0.0:%d", PORT)

    clients: list[_ClientHandler] = []
    running = True

    # Hilo que acepta conexiones entrantes
    def _accept_loop():
        nonlocal running
        while running:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            handler = _ClientHandler(conn, addr, init_msg)
            handler.start()
            clients.append(handler)
        try:
            srv.close()
        except Exception:
            pass

    accept_thread = threading.Thread(target=_accept_loop, daemon=True)
    accept_thread.start()

    # ---- Bucle principal: despachar colas ------------------------ #
    while running:
        # 1) Cola de control (acciones / epochs) — no bloqueante
        while True:
            try:
                msg = control_queue.get_nowait()
            except Exception:
                break
            if msg is _SENTINEL:
                running = False
                break
            if "action" in msg:
                pkt = {"type": "action", "text": msg["action"]}
            elif "epoch" in msg:
                pkt = {
                    "type": "epoch",
                    "current": int(msg["epoch"]),
                    "total": int(msg.get("total", total_epochs)),
                }
            else:
                continue
            _broadcast(clients, pkt)

        if not running:
            break

        # 2) Cola de datos EEG
        #    Si no hay clientes conectados, vaciamos la cola sin enviar
        #    para evitar que crezca indefinidamente.
        while True:
            try:
                chunk = data_queue.get_nowait()
            except Exception:
                break
            if chunk is _SENTINEL:
                running = False
                break
            _purge_dead(clients)
            if not clients:
                continue  # descartar
            pkt = {"type": "data", "samples": _ndarray_to_lists(chunk)}
            _broadcast(clients, pkt)

        if not running:
            break

        # Pequeña espera para no hacer busy-wait
        import time
        time.sleep(0.004)

    # ---- Limpieza ------------------------------------------------ #
    for c in clients:
        c.alive = False
    try:
        srv.close()
    except Exception:
        pass
    log.info("Servidor detenido.")


# ------------------------------------------------------------------ #
#  Utilidades internas
# ------------------------------------------------------------------ #
def _ndarray_to_lists(arr) -> list[list[float]]:
    """Convierte un ndarray o lista a lista de listas."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def _purge_dead(clients: list[_ClientHandler]):
    """Elimina clientes desconectados de la lista."""
    clients[:] = [c for c in clients if c.alive]


def _broadcast(clients: list[_ClientHandler], pkt: dict):
    """Envía un paquete a todos los clientes vivos."""
    _purge_dead(clients)
    for c in clients:
        c.send(pkt)


# ------------------------------------------------------------------ #
#  Punto de entrada para multiprocessing.Process
# ------------------------------------------------------------------ #
def run_server(
    ch_names: Sequence[str],
    ch_types: str | Sequence[str],
    sfreq: float,
    total_epochs: int,
    initial_action: str,
    data_queue: Queue,
    control_queue: Queue,
):
    """
    Función de entrada para lanzar el servidor como un ``Process``.

    Parameters
    ----------
    ch_names : list[str]
        Nombres de canales EEG (se envía en el paquete ``init``).
    ch_types : str | list[str]
        Tipo(s) de canal (``"eeg"``, lista por canal, etc.).
    sfreq : float
        Frecuencia de muestreo.
    total_epochs : int
        Número total de epochs esperados.
    initial_action : str
        Texto de la acción inicial (p.ej. ``"Reposo:"``).
    data_queue : Queue
        Cola de chunks EEG ``np.ndarray (n_ch, n_samples)``.
        Si no hay clientes conectados los datos se descartan.
    control_queue : Queue
        Cola de comandos:
        - ``{"action": "texto"}`` → cambia la acción en el visor.
        - ``{"epoch": N, "total": M}`` → actualiza barra de progreso.
        - ``None`` → detiene el servidor.
    """
    _server_loop(
        ch_names=list(ch_names),
        ch_types=ch_types if isinstance(ch_types, str) else list(ch_types),
        sfreq=sfreq,
        total_epochs=total_epochs,
        initial_action=initial_action,
        data_queue=data_queue,
        control_queue=control_queue,
    )

class EEGLiveServer:
    def __init__(
        self,
        ch_names: Sequence[str],
        sfreq: float,
        total_epochs: int = 0,
        ch_types: str | Sequence[str] = "eeg",
        initial_action: str = "Reposo:",
    ):
        self.ch_names = list(ch_names)
        self.ch_types = ch_types
        self.sfreq = sfreq
        self.total_epochs = total_epochs
        self.initial_action = initial_action
        self.current_epoch = 0

        self.data_queue: Queue = MPQueue()
        self.control_queue: Queue = MPQueue()
        self.p: Process | None = None

    def start(self):
        self.p = Process(
            target=run_server,
            args=(
                self.ch_names,
                self.ch_types,
                self.sfreq,
                self.total_epochs,
                self.initial_action,
                self.data_queue,
                self.control_queue,
            ),
        )
        self.p.start()

    def newChunk(self, chunk: np.ndarray, chunk_size: int):
        self.data_queue.put(chunk)

    def setAction(self, action: str):
        self.control_queue.put({"action": action})

    def increaseEpoch(self):
        self.current_epoch += 1
        self.control_queue.put({"epoch": self.current_epoch, "total": self.total_epochs})

    def set_epoch(self, current_epoch: int, total_epochs: int):
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        self.control_queue.put({"epoch": self.current_epoch, "total": self.total_epochs})

    def stop(self):
        self.control_queue.put(None)
        if self.p is not None:
            self.p.join(timeout=3)

# ------------------------------------------------------------------ #
#  Ejecución directa (demo)
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    """
    Lanza el servidor con datos aleatorios para probar con el visor.
    """
    import time
    from multiprocessing import Process, Queue as MPQueue

    ch_names = [
        "Fp1", "Fp2", "F3", "Fz", "F4",
        "C3", "Cz", "C4", "P3", "Pz",
        "P4", "O1", "O2", "T7", "T8",
    ]
    sfreq = 250.0
    total_epochs = 40

    dq: Queue = MPQueue()
    cq: Queue = MPQueue()

    p = Process(
        target=run_server,
        args=(ch_names, "eeg", sfreq, total_epochs, "Reposo:", dq, cq),
    )
    p.start()

    print(f"Servidor lanzado en puerto {PORT}.  Ejecuta eeg_live_pyqtgraph.py para conectar.")
    print("Generando datos aleatorios...  Ctrl+C para parar.\n")

    epoch = 0
    try:
        while True:
            # Enviar ~250 muestras/s en chunks de 10
            chunk = np.random.randn(len(ch_names), 10) * 20
            dq.put(chunk)
            time.sleep(10 / sfreq)

            # Simular epoch cada 5 s
            if np.random.rand() < (10 / sfreq) / 5:
                epoch = min(epoch + 1, total_epochs)
                cq.put({"epoch": epoch, "total": total_epochs})
                if epoch % 5 == 0:
                    actions = ["Mano izq:", "Mano der:", "Pies:", "Reposo:"]
                    cq.put({"action": actions[epoch % len(actions)]})
    except KeyboardInterrupt:
        cq.put(None)
        p.join(timeout=3)
        print("\nServidor detenido.")

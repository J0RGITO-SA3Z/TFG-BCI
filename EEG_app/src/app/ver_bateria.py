import os, sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager
from rich.panel import Panel
from rich.text import Text

from app.app_utils import seleccionarPuertoCOM


def ver_bateria(console):
    console.clear()
    console.print(Panel("[bold]Estado de batería del EEG[/bold]", border_style="cyan"))

    puerto_COM = seleccionarPuertoCOM(console)
    if puerto_COM is None:
        return

    console.print(f"Conectando a [cyan]{puerto_COM}[/cyan]...")

    eeg = acquisition.EEG()  # inicia bacore internamente
    try:
        with EEGManager() as mgr:
            eeg.setup(mgr, port=puerto_COM, cap={})
            batt = mgr.get_battery_info()
            mgr.disconnect()
    finally:
        eeg.close()  # cierra bacore

    nivel = batt.level
    cargando = batt.is_charging
    cargador = batt.is_charger_connected

    if nivel >= 60:
        color = "green"
    elif nivel >= 25:
        color = "yellow"
    else:
        color = "red"

    texto = Text()
    texto.append("  Nivel:    ", style="bold")
    texto.append(f"{nivel}%\n", style=f"bold {color}")
    texto.append("  Cargador: ", style="bold")
    texto.append("Conectado\n" if cargador else "No conectado\n")
    texto.append("  Estado:   ", style="bold")
    texto.append("Cargando" if cargando else "En uso")

    console.print(Panel(texto, border_style=color, title="Batería"))
    console.input("\n[dim]Pulse Enter para continuar...[/dim]")

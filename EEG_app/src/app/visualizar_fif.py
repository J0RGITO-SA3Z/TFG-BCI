import os, sys
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import mne
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.app_utils import RECORD_DIR

_EXPERIMENTOS = {
    "1": "experimento_visual",
    "2": "experimento_juegos",
}


def _listar_sujetos(exp_dir: Path) -> list[Path]:
    if not exp_dir.exists():
        return []
    return sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("suj")], key=lambda d: d.name)


def _listar_fifs(suj_dir: Path) -> list[Path]:
    if not suj_dir.exists():
        return []
    return sorted(suj_dir.glob("*_raw.fif"), key=lambda p: p.name.lower())


def visualizar_fif(console: Console) -> None:
    console.clear()
    console.print(Panel(
        "[bold]Visualizar FIFs[/bold]\n"
        "Selecciona experimento, sujeto y archivo .fif para visualizar.",
        border_style="cyan",
        padding=(1, 2),
    ))

    console.print("  [cyan]1[/cyan] Experimento Visual")
    console.print("  [cyan]2[/cyan] Experimento Juegos")
    console.print("  [cyan]-1[/cyan] Cancelar\n")
    exp_choice = console.input(">> ").strip()

    if exp_choice == "-1":
        return
    if exp_choice not in _EXPERIMENTOS:
        console.print("[red]Opción no válida[/red]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return

    exp_name = _EXPERIMENTOS[exp_choice]
    exp_dir = RECORD_DIR / exp_name

    sujetos = _listar_sujetos(exp_dir)
    if not sujetos:
        console.print(f"[red]No se encontraron sujetos en {exp_dir}[/red]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return

    suj_table = Table(title=f"Sujetos — {exp_name}", expand=True)
    suj_table.add_column("ID", style="cyan", justify="right")
    suj_table.add_column("Sujeto")
    for idx, suj in enumerate(sujetos):
        suj_table.add_row(str(idx + 1), suj.name)
    console.print(suj_table)
    console.print("\n  [cyan]-1[/cyan] Cancelar")

    suj_input = console.input("Selecciona sujeto (ID o nombre) >> ").strip()
    if suj_input == "-1":
        return

    suj_dir = None
    if suj_input.isdigit():
        idx = int(suj_input) - 1
        if 0 <= idx < len(sujetos):
            suj_dir = sujetos[idx]
    if suj_dir is None:
        matches = [s for s in sujetos if suj_input.lower() in s.name.lower()]
        if len(matches) == 1:
            suj_dir = matches[0]
        elif len(matches) > 1:
            console.print(f"[red]Múltiples coincidencias. Sé más específico.[/red]")
            console.input("[dim]Pulse Enter para continuar...[/dim]")
            return

    if suj_dir is None:
        console.print("[red]Sujeto no encontrado[/red]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return

    fifs = _listar_fifs(suj_dir)
    if not fifs:
        console.print(f"[red]No se encontraron archivos .fif para {suj_dir.name}[/red]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return

    fif_table = Table(title=f"Grabaciones — {suj_dir.name}", expand=True)
    fif_table.add_column("ID", style="cyan", justify="right")
    fif_table.add_column("Archivo")
    for idx, fif in enumerate(fifs):
        fif_table.add_row(str(idx + 1), fif.name)
    console.print(fif_table)
    console.print("\n  [cyan]-1[/cyan] Cancelar")

    fif_input = console.input("Selecciona archivo (ID) >> ").strip()
    if fif_input == "-1":
        return

    if not fif_input.isdigit() or not (1 <= int(fif_input) <= len(fifs)):
        console.print("[red]Selección no válida[/red]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return

    fif_path = fifs[int(fif_input) - 1]
    console.print(f"[cyan]Cargando {fif_path.name}...[/cyan]")

    try:
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        raw.filter(1, 40, verbose=False).plot(scalings='auto', verbose=False)
    except Exception as e:
        console.print(f"[red]Error al cargar/visualizar el archivo: {e}[/red]")

    console.input("[dim]Pulse Enter para continuar...[/dim]")

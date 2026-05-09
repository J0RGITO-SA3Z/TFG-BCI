import os, sys
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from app.app_utils import RECORD_DIR, lee_indice
from offline_tests.pipelineEval import run_MiRepNet_pipeline

EXPERIMENTO_VISUAL_DIR = RECORD_DIR / "experimento_visual"


def _listar_sujetos(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name.lower(),
    )


def _listar_fifs(suj_dir: Path) -> list[Path]:
    return sorted(suj_dir.glob("*.fif"), key=lambda p: p.name.lower())


def _mostrar_sujetos(console, sujetos: list[Path]) -> None:
    table = Table(title="Sujetos disponibles — experimento_visual", expand=True)
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Carpeta")
    table.add_column("Grabaciones .fif", justify="right")

    if not sujetos:
        table.add_row("-", "[italic dim]No se encontraron carpetas de sujetos[/]", "-")
    else:
        for idx, suj in enumerate(sujetos):
            n_fifs = len(list(suj.glob("*.fif")))
            table.add_row(str(idx), suj.name, str(n_fifs))

    console.print(table)


def _mostrar_fifs(console, fifs: list[Path], suj_name: str) -> None:
    table = Table(title=f"Grabaciones — {suj_name}", expand=True)
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Archivo")

    if not fifs:
        table.add_row("-", "[italic dim]No se encontraron archivos .fif[/]")
    else:
        for idx, fif in enumerate(fifs):
            table.add_row(str(idx), fif.name)

    console.print(table)


def _seleccionar_sujeto(console) -> Path | None:
    sujetos = _listar_sujetos(EXPERIMENTO_VISUAL_DIR)

    console.clear()
    console.print(
        Panel(
            "[bold]Evaluacion de sujeto con MiRepNet[/bold]\n"
            "Seleccione la carpeta del sujeto a evaluar.",
            border_style="white",
            padding=(1, 2),
        )
    )
    _mostrar_sujetos(console, sujetos)

    if not sujetos:
        console.print("[red]No hay carpetas de sujetos en experimento_visual.[/red]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return None

    indice = lee_indice(
        console.input("\nSeleccione ID de sujeto [dim](-1 para cancelar)[/dim]: ")
    )

    while indice != -1 and (indice < 0 or indice >= len(sujetos)):
        indice = lee_indice(
            console.input("[red]Índice invalido. Intente de nuevo:[/red] ")
        )

    if indice == -1:
        return None

    return sujetos[indice]


def _seleccionar_fifs(console, suj_dir: Path) -> list[str]:
    fifs = _listar_fifs(suj_dir)

    console.clear()
    console.print(
        Panel(
            f"[bold]Evaluacion de sujeto con MiRepNet — {suj_dir.name}[/bold]\n"
            "Introduzca los IDs separados por comas o [cyan]all[/cyan] para seleccionar todos.",
            border_style="white",
            padding=(1, 2),
        )
    )
    _mostrar_fifs(console, fifs, suj_dir.name)

    if not fifs:
        console.print("[red]No hay archivos .fif en esta carpeta.[/red]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return []

    entrada = console.input(
        "\nIDs de grabaciones [dim](ej: 0,2 · all · -1 para cancelar)[/dim]: "
    ).strip()

    if entrada == "-1":
        return []

    if entrada.lower() == "all":
        return [str(f.resolve()) for f in fifs]

    paths: list[str] = []
    for tok in entrada.split(","):
        tok = tok.strip()
        if not tok:
            continue
        idx = lee_indice(tok)
        if 0 <= idx < len(fifs):
            paths.append(str(fifs[idx].resolve()))
        else:
            console.print(f"[yellow]ID {tok!r} ignorado (fuera de rango).[/yellow]")

    return paths


def evaluar_sujeto_MiRepNet(console) -> None:
    suj_dir = _seleccionar_sujeto(console)
    if suj_dir is None:
        return

    fif_paths = _seleccionar_fifs(console, suj_dir)
    if not fif_paths:
        return

    console.print("\n[cyan]Archivos seleccionados:[/cyan]")
    for p in fif_paths:
        console.print(f"  - {Path(p).name}")

    console.print("\n[yellow]Ejecutando evaluacion MiRepNet...[/yellow]")
    try:
        run_MiRepNet_pipeline(fif_paths)
        console.print("[green]Evaluacion finalizada correctamente.[/green]")
    except Exception as e:
        console.print(f"[red]Error durante la evaluacion: {e}[/red]")

    console.input("[dim]Pulse Enter para continuar...[/dim]")

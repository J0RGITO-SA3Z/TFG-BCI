from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from pipeline_RT.pipelineEval import run_MiRepNet_pipeline


def _listar_fifs_disponibles(recordings_dir: Path) -> list[Path]:
	if not recordings_dir.exists():
		return []
	return sorted(recordings_dir.glob("*.fif"), key=lambda p: p.name.lower())


def _mostrar_fifs_disponibles(console, fifs: list[Path]) -> None:
	table = Table(title="Grabaciones disponibles", expand=True)
	table.add_column("ID", style="cyan", justify="right")
	table.add_column("Archivo")

	if not fifs:
		table.add_row("-", "[italic dim]No se encontraron archivos .fif en recordings[/]")
	else:
		for idx, fif in enumerate(fifs):
			table.add_row(str(idx), fif.name)

	console.print(table)


def _resolver_fif_paths_desde_entrada(entrada: str, recordings_dir: Path, fifs_disponibles: list[Path]) -> list[str]:
	entrada = entrada.strip()
	if not entrada:
		return []

	# Si la entrada incluye ".fif" o comas, se interpreta como lista de archivos.
	if ".fif" in entrada.lower() or "," in entrada:
		tokens = [tok.strip() for tok in entrada.split(",") if tok.strip()]
		paths: list[str] = []

		for tok in tokens:
			candidato = Path(tok)
			if not candidato.is_absolute():
				candidato = recordings_dir / tok

			candidato = candidato.resolve()
			if candidato.exists() and candidato.suffix.lower() == ".fif":
				paths.append(str(candidato))

		return paths

	# Si no, se interpreta como nombre (o fragmento) de sujeto.
	sujeto = entrada.lower()
	return [str(p.resolve()) for p in fifs_disponibles if sujeto in p.name.lower()]


def evaluar_sujeto_MiRepNet(console) -> None:
	recordings_dir = Path(__file__).resolve().parents[1] / "recordings"
	fifs_disponibles = _listar_fifs_disponibles(recordings_dir)

	console.clear()
	console.print(
		Panel(
			"[bold]Evaluacion de sujeto con MiRepNet[/bold]\n"
			"Introduce una lista de archivos .fif separada por comas\n"
			"o un nombre de sujeto para buscar coincidencias en recordings.",
			border_style="white",
			padding=(1, 2),
		)
	)
	_mostrar_fifs_disponibles(console, fifs_disponibles)

	entrada = Prompt.ask(
		"Archivos .fif (coma separados) o nombre de sujeto [dim](-1 para cancelar)[/dim]"
	).strip()

	if entrada == "-1":
		return

	fif_paths = _resolver_fif_paths_desde_entrada(entrada, recordings_dir, fifs_disponibles)
	if not fif_paths:
		console.print("[red]No se encontraron archivos .fif validos para esa entrada.[/red]")
		console.input("[dim]Pulse Enter para continuar...[/dim]")
		return

	console.print("[cyan]Archivos seleccionados:[/cyan]")
	for p in fif_paths:
		console.print(f"  - {p}")

	console.print("[yellow]Ejecutando evaluacion MiRepNet...[/yellow]")
	try:
		run_MiRepNet_pipeline(fif_paths)
		console.print("[green]Evaluacion finalizada correctamente.[/green]")
	except Exception as e:
		console.print(f"[red]Error durante la evaluacion: {e}[/red]")

	console.input("[dim]Pulse Enter para continuar...[/dim]")

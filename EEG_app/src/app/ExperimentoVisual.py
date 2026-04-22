import os, sys
import json
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.panel import Panel

from app.app_utils import seleccionarPuertoCOM
from app.app_utils import RECORD_DIR
from app.medicion_impedancias import build_channels_table
from app.visualizar_impedancias import visualizar_impedancias
from components.EEG_BrainAccess.ImpedanceMeter import ImpedanceMeter
from components.DataProvider.ExperimentoVisualDataProvider import ExperimentoVisualDataProvider

class ExperimentoVisual:

    def __init__(self, console: Console):
        self.console = console
        return

    def _next_file_path(self, suj_num: int):
        suj_dir = RECORD_DIR / "experimento_visual" / f"suj{suj_num}"
        suj_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(suj_dir.glob(f"suj{suj_num}_*_raw.fif"))
        if not existing:
            next_idx = 1
        else:
            last = existing[-1].stem
            parts = last.split("_")
            try:
                next_idx = int(parts[1]) + 1
            except (IndexError, ValueError):
                next_idx = len(existing) + 1

        return suj_dir / f"suj{suj_num}_{next_idx}_raw.fif"

    def _cuestionario(self, suj_num: int, suj_dir):
        self.console.clear()
        self.console.print(Panel(
            f"[bold]Cuestionario inicial — Sujeto {suj_num}[/bold]\n"
            "[dim]Este cuestionario solo se realiza la primera vez.[/dim]",
            border_style="cyan"
        ))

        def ask_opcion(pregunta, opciones):
            etiquetas = " / ".join(f"[cyan]{i+1}[/cyan] {o}" for i, o in enumerate(opciones))
            while True:
                self.console.print(f"\n{pregunta}")
                self.console.print(etiquetas)
                resp = self.console.input(">> ").strip()
                if resp.isdigit() and 1 <= int(resp) <= len(opciones):
                    return opciones[int(resp) - 1]
                self.console.print("[red]Opción no válida[/red]")

        edad = None
        while edad is None:
            try:
                edad = int(self.console.input("\nEdad: ").strip())
            except ValueError:
                self.console.print("[red]Introduce un número válido[/red]")

        sexo = ask_opcion("Sexo:", ["Masculino", "Femenino", "Otro", "Prefiero no decirlo"])
        dominancia = ask_opcion("Dominancia manual:", ["Diestro", "Zurdo", "Ambidiestro"])
        densidad = ask_opcion("Densidad capilar:", ["Sin cabello", "Cabello muy escaso", "Cabello corto y fino", "Cabello corto y grueso", "Cabello largo y fino", "Cabello largo y grueso"])
        experiencia_bci = ask_opcion("¿Tiene experiencia previa con BCI?", ["Ninguna", "Poca (1-2 sesiones)", "Moderada (3-10 sesiones)", "Alta (>10 sesiones)"])
        problemas_neurologicos = ask_opcion("¿Tiene algún problema neurológico conocido?", ["No", "Sí"])
        if problemas_neurologicos == "Sí":
            detalle = self.console.input("Indique cuál: ").strip()
            problemas_neurologicos = detalle if detalle else "Sí (sin especificar)"

        datos = {
            "sujeto": suj_num,
            "edad": edad,
            "sexo": sexo,
            "dominancia_manual": dominancia,
            "densidad_capilar": densidad,
            "experiencia_bci": experiencia_bci,
            "problemas_neurologicos": problemas_neurologicos,
        }

        perfil_path = suj_dir / f"suj{suj_num}_perfil.json"
        with open(perfil_path, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)

        self.console.print(f"\n[green]Perfil guardado en {perfil_path}[/green]")
        self.console.input("[dim]Pulse Enter para continuar...[/dim]")

    def _prueba_impedancias(self, puerto_COM, channelsConfig):
        """Mide impedancias, muestra tabla + gráfico y permite repetir. Devuelve el dict {electrodo: valor}."""
        imp_raw = None

        while True:
            self.console.clear()
            self.console.print(Panel("[bold]Prueba de impedancias[/bold]", border_style="cyan"))

            meter = ImpedanceMeter(puerto_com=puerto_COM, canales=channelsConfig)
            imp_raw = meter.measure(console=self.console)

            self.console.clear()
            self.console.print(build_channels_table(channelsConfig, imp_raw))

            imp_map = meter.to_map()
            visualizar_impedancias(electrodos=imp_map, umbral_bueno=10, umbral_malo=100)

            self.console.print("\n[bold]Opciones:[/bold]")
            self.console.print("  [cyan]1[/cyan] Repetir medición")
            self.console.print("  [cyan]2[/cyan] Continuar con el experimento")

            opcion = self.console.input(">> ").strip()
            while opcion not in ("1", "2"):
                self.console.print("[red]Opción no válida[/red]")
                opcion = self.console.input(">> ").strip()

            if opcion == "2":
                return imp_map

    def _guardar_impedancias_en_perfil(self, suj_num: int, suj_dir, sesion_idx: int, imp_map: dict):
        perfil_path = suj_dir / f"suj{suj_num}_perfil.json"
        if not perfil_path.exists():
            return

        with open(perfil_path, "r", encoding="utf-8") as f:
            perfil = json.load(f)

        perfil[f"impedancias_sesion_{sesion_idx}"] = imp_map

        with open(perfil_path, "w", encoding="utf-8") as f:
            json.dump(perfil, f, indent=2, ensure_ascii=False)

    def start(self, channelsConfig):
        """Ejecuta experimento visual usando ExperimentoVisualDataProvider."""
        self.console.clear()
        self.console.print("En el modo experimento visual, las acciones anotadas siempre son izquierda, derecha, abajo, descanso. Si quieres anotaciones personalizadas, usa el modo de grabación manual.")
        self.console.input("Pulse Enter para continuar...")
        self.console.clear()

        puerto_COM = seleccionarPuertoCOM(self.console)

        if puerto_COM is None:
            return

        self.console.print("[green]EEG configurado correctamente[/green]\n")

        num_trials_clase = IntPrompt.ask("Introduce el numero de trials por clase (4 clases y 8s por trial)")
        suj_num = IntPrompt.ask("Número de sujeto")

        suj_dir = RECORD_DIR / "experimento_visual" / f"suj{suj_num}"
        es_nuevo = not suj_dir.exists()
        fileOutput = self._next_file_path(suj_num)
        sesion_idx = int(fileOutput.stem.split("_")[1])

        if es_nuevo:
            self._cuestionario(suj_num, suj_dir)

        imp_map = self._prueba_impedancias(puerto_COM, channelsConfig)

        self.console.clear()
        self.console.print(f"Archivo de salida: [cyan]{fileOutput}[/cyan]")

        data_provider = ExperimentoVisualDataProvider(
            puerto_COM=puerto_COM,
            channelsConfig=channelsConfig,
            numTrialsClase=num_trials_clase,
            lista=["IZQUIERDA", "DERECHA", "ABAJO"],
        )

        data_provider.get_data(fif_path=fileOutput)
        raw = data_provider.get_raw()

        self._guardar_impedancias_en_perfil(suj_num, suj_dir, sesion_idx, imp_map)

        self.console.clear()
        self.console.print(f"\nEEG grabado y guardado en [green]{fileOutput}[/green]")
        raw.filter(1, 40).plot(scalings='auto', verbose=False)
        self.console.input("[dim]Pulse Enter para continuar...[/dim]")

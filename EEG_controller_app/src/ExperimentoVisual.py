from rich.console import Console
from rich.prompt import IntPrompt

from app_utils import seleccionarPuertoCOM
from app_utils import RECORD_DIR
from pipeline_RT.DataProvider.ExperimentoVisualDataProvider import ExperimentoVisualDataProvider

class ExperimentoVisual:

    def __init__(self, console: Console):
        self.console = console
        return

    def start(self,channelsConfig):
        """Ejecuta experimento visual usando ExperimentoVisualDataProvider."""
        self.console.clear()
        self.console.print("En el modo experimento visual, las acciones anotadas siempre son hizquierda, derecha, abajo, descanso. Si quieres anotaciones personalizadas, usa el modo de grabación manual.")
        self.console.input("Pulse Enter para continuar...")
        self.console.clear()

        puerto_COM = seleccionarPuertoCOM(self.console)

        if puerto_COM is None:
            return

        self.console.print("[green]EEG configurado correctamente[/green]\n")

        num_trials_clase = IntPrompt.ask("Introduce el numero de trials por clase (4 clases y 8s por trial)")
        entrada = self.console.input("Introduce el nombre del archivo de salida (sin extensión): ").strip()
        fileOutput = RECORD_DIR / f"{entrada}.fif"

        data_provider = ExperimentoVisualDataProvider(
            puerto_COM=puerto_COM,
            channelsConfig=channelsConfig,
            numTrialsClase=num_trials_clase,
            lista=["IZQUIERDA", "DERECHA", "ABAJO", "DESCANSO"],
        )

        data_provider.get_data(fif_path=fileOutput)
        raw = data_provider.get_raw()

        self.console.clear()
        self.console.print(f"\nEEG grabado y guardado en [green]{fileOutput}[/green]")
        raw.filter(1, 40).plot(scalings='auto', verbose=False)
        self.console.input("[dim]Pulse Enter para continuar...[/dim]")
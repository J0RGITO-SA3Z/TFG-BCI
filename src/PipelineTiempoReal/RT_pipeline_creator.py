import os, sys
from Training_real_time import Training_real_time
from rich.console import Console
from brainaccess.core.eeg_manager import EEGManager

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ── Piezas del script ─────────────────────────────────────────────────────────────────────

from PipelineTiempoReal.RT_pipeline_process import RT_pipeline
from real_time_emulator import RealTimeEmulator
from utils.channels_config import load_channels_conf
from utils.EEGRecorder import EEGRecorder

def main():
    console = Console()
    trainer = Training_real_time(console)
    channels_config = load_channels_conf(sys.argv[1])

    matriz,modelo  = trainer.start(channels_config)
    eeg = EEGRecorder() 

    with EEGManager() as mgr:
        eeg.configAndConect(mgr=mgr, COM_port=puerto_COM, channelConfig=channelsConfig, gain=8)

if __name__ == "__main__":
    main()
import os, sys

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ── Piezas del script ─────────────────────────────────────────────────────────────────────

from RT_pipeline import RT_pipeline
from real_time_emulator import RealTimeEmulator

def main():
    print("Pipeline creator for real-time processing")

if __name__ == "__main__":
    main()
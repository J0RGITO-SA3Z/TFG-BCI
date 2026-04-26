"""
Exporta MiRepNet a formato ONNX para visualizarlo con Netron.

Uso:
    python export_mirepnet_onnx.py

Genera: mirepnet.onnx en el mismo directorio.
Para visualizar: abre mirepnet.onnx en https://netron.app  (o con `netron mirepnet.onnx`)

Librerías necesarias:
    pip install onnx
    pip install netron  (opcional, solo si quieres lanzarlo desde aquí)
"""

import os
import sys
import torch
import torch.nn as nn

# ── Rutas ──────────────────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT     = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_ROOT, "..", ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
OUTPUT_PATH  = os.path.join(THIS_DIR, "mirepnet.onnx")

for _p in [PROJECT_ROOT, SRC_ROOT, MIREPNET_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components.pretrainedModels.MiRepNet.model.mlm import mlm_mask

# ── Dimensiones de entrada ─────────────────────────────────────────────────────
# El modelo espera (batch, canales, tiempo).
# Tras SpatialInterpolator siempre hay 45 canales.
# 1000 muestras = 4 s a 250 Hz (valor típico del pipeline).
N_CHANNELS   = 45
N_TIME       = 1000
N_CLASSES    = 2          # left_hand / right_hand


# ── Wrapper que expone solo la salida de clasificación ─────────────────────────
# mlm_mask.forward() devuelve (pooled, cls_logits); ONNX necesita un único tensor.
class MiRepNetExportWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, cls_output = self.model(x)
        return cls_output


def export():
    device = torch.device("cpu")   # ONNX se exporta siempre en CPU

    print("Cargando modelo...")
    base = mlm_mask(
        emb_size=256,
        depth=6,
        n_classes=N_CLASSES,
        pretrainmode=False,
        pretrain=WEIGHT_PATH,
    ).to(device)
    base.eval()

    model = MiRepNetExportWrapper(base).to(device)
    model.eval()

    # Entrada ficticia con las dimensiones reales del pipeline
    dummy_input = torch.randn(1, N_CHANNELS, N_TIME, device=device)

    print(f"Exportando a ONNX → {OUTPUT_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["eeg"],          # nombre visible en Netron
        output_names=["logits"],
        dynamic_axes={
            "eeg":    {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print("Exportación completada.")

    # ── Validación con onnx (si está instalado) ────────────────────────────────
    try:
        import onnx
        model_onnx = onnx.load(OUTPUT_PATH)
        onnx.checker.check_model(model_onnx)
        print("Validación ONNX: OK")
    except ImportError:
        print("(instala 'onnx' para validar el fichero exportado)")

    # ── Abrir con Netron (si está instalado) ───────────────────────────────────
    try:
        import netron
        print("Abriendo Netron...")
        netron.start(OUTPUT_PATH)
    except ImportError:
        print("\nPara visualizar el modelo:")
        print(f"  · Opción 1 (web):   arrastra '{OUTPUT_PATH}' a https://netron.app")
        print(f"  · Opción 2 (local): pip install netron  →  netron mirepnet.onnx")


if __name__ == "__main__":
    export()

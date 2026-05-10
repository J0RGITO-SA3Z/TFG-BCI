import os, sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MIREPNET_DIR not in sys.path:
    sys.path.append(MIREPNET_DIR)

from components.pretrainedModels.MiRepNet.utils.utils import *


import numpy as np
import matplotlib.pyplot as plt
import mne

montage = mne.channels.make_standard_montage("standard_1020")
channel_names = montage.ch_names

info = mne.create_info(
    ch_names=channel_names,
    sfreq=100,        # valor dummy
    ch_types="eeg"
)

exclude = ["T3", "T4", "T5", "T6"]

info.set_montage(montage)
layout = mne.channels.find_layout(info, exclude=exclude)

# diferencias en mayusculas
nombresMNE = [n.upper() for n in layout.names]
nombresMiRepNet = all_channels_names

canales_faltantes_repnet = []
for n in nombresMNE:
    if n not in nombresMiRepNet:
        print("Canal en MNE no en MiRepNet:", n)
        canales_faltantes_repnet.append(n)

canales_faltantes_MNE = []
for n in nombresMiRepNet:
    if n not in nombresMNE:
        print("Canal en MiRepNet no en MNE:", n)
        canales_faltantes_MNE.append(n)

posicionesUsadas = np.array(
    [
        pos
        for name, pos in channel_positions.items()
        if name.upper() in {c.upper() for c in use_channels_names}
    ]
)

objetivo_upper = {c.upper() for c in canales_faltantes_repnet}
mi_idx = [layout.names.index(n) for n in layout.names if n.upper() in objetivo_upper]

objetivo_upper2 = {c.upper() for c in use_channels_names}
excluir_restantes2 = [n for n in layout.names if n.upper() not in objetivo_upper2] + exclude
layout3 = mne.channels.find_layout(info, exclude=excluir_restantes2)

posiciones = np.array([channel_positions[name] for name in all_channels_names])

miRep_x = posiciones[:, 0]
miRep_y = posiciones[:, 1]

miRep_x3 = posicionesUsadas[:, 0]
miRep_y3 = posicionesUsadas[:, 1]

x = layout.pos[:, 0]
y = layout.pos[:, 1]

x2 = [layout.pos[i, 0] for i in mi_idx]
y2 = [layout.pos[i, 1] for i in mi_idx]

x3 = layout3.pos[:, 0]
y3 = layout3.pos[:, 1]

print("Canales MNE:", len(x))
print("Canales MiRepNet:", len(miRep_x))

print("nombres MNE:", layout.names)
print("nombres MiRepNet:", list(channel_positions.keys()))

# Ventana MNE
fig_mne, axes_mne = plt.subplots(1, 2, figsize=(12, 5))
fig_mne.suptitle("Layouts MNE")

axes_mne[0].scatter(x, y, s=40, color="tab:blue")
axes_mne[0].scatter(x2, y2, s=40, color="tab:blue")
axes_mne[0].set_aspect("equal")
axes_mne[0].axis("off")
axes_mne[0].set_title("Layout 2D MNE")

axes_mne[1].scatter(x3, y3, s=40, color="tab:blue")
axes_mne[1].set_aspect("equal")
axes_mne[1].axis("off")
axes_mne[1].set_title("Layout 2D MNE de electrodos usados por el modelo MiRepNet")

fig_mne.tight_layout()

# Ventana MiRepNet
fig_rep, axes_rep = plt.subplots(1, 2, figsize=(12, 5))
fig_rep.suptitle("Layouts MiRepNet")

axes_rep[0].scatter(miRep_x, miRep_y, s=40, color="tab:red")
axes_rep[0].set_aspect("equal")
axes_rep[0].axis("off")
axes_rep[0].set_title("Layout 2D MiRepNet")

axes_rep[1].scatter(miRep_x3, miRep_y3, s=40, color="tab:red")
axes_rep[1].set_aspect("equal")
axes_rep[1].axis("off")
axes_rep[1].set_title("Layout 2D MiRepNet de electrodos usados por el modelo MiRepNet")

fig_rep.tight_layout()

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from channel_list import *
import mne

montage = mne.channels.make_standard_montage("standard_1005")
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

fig, axes = plt.subplots(2, 2, figsize=(12, 6))

# MNE
axes[0][0].scatter(x, y, s=40, label="Canales en MiRepNet")
axes[0][0].scatter(x2, y2, s=40, color="tab:red", label="Canales no en MiRepNet")
axes[0][0].set_aspect("equal")
axes[0][0].axis("off")
axes[0][0].set_title("Layout 2D con todos los electrodos – MNE")

# MiRepNet
axes[0][1].scatter(miRep_x, miRep_y, s=40, color="tab:red")
axes[0][1].set_aspect("equal")
axes[0][1].axis("off")
axes[0][1].set_title("Layout 2D con todos los electrodos – MiRepNet")

# MNE Usados
axes[1][0].scatter(x3, y3, s=40)
axes[1][0].set_aspect("equal")
axes[1][0].axis("off")
axes[1][0].set_title("Layout 2D de electrodos usados por MiRepNet – MNE")

# Mi RepNet Usados
axes[1][1].scatter(miRep_x3, miRep_y3, s=40, color="tab:red")
axes[1][1].set_aspect("equal")
axes[1][1].axis("off")
axes[1][1].set_title("Layout 2D de electrodos usados por MiRepNet - MiRepNet")

plt.tight_layout()
plt.show()

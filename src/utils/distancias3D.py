"""
Compute 3D Euclidean distances between your EEG headset channels
and the 45-channel MIRepNet template using the standard 10-20 montage (MNE).
Author: Jorge + ChatGPT (2025)
"""

import os
import json
import numpy as np
from scipy.spatial.distance import cdist
import mne
import matplotlib
matplotlib.use('TkAgg')  # o 'Qt5Agg' si tienes PyQt5 instalado

import matplotlib.pyplot as plt


# Cargamos el array de canales desde los archivos JSON en la carpeta Disposition

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(SRC_ROOT, "Disposition", "configs")

# ------------------------------------------------------------
# 1. Your headset (BrainAccess Cap or similar)
# ------------------------------------------------------------

HEADSET_DIR = os.path.join(CONFIG_DIR, "channels_headset_15.json")
headset_file = open (HEADSET_DIR)
with open(HEADSET_DIR, "r") as f:
    cfg = json.load(f)

your_channels = cfg["channels"]

# ------------------------------------------------------------
# 2. MIRepNet 45-channel template (regions FC, C, CP, T)
# ------------------------------------------------------------

TEMPLATE_DIR = os.path.join(CONFIG_DIR, "channels_template_45.json")
headset_file = open (HEADSET_DIR)
template_file = open (TEMPLATE_DIR)

with open(TEMPLATE_DIR, "r") as d:
    cfg = json.load(d)

template_45 = cfg["channels"]

# ------------------------------------------------------------
# 3. Load 10-20 montage from MNE (3D coordinates)
# ------------------------------------------------------------
montage = mne.channels.make_standard_montage('standard_1020')
ch_pos = montage.get_positions()['ch_pos']
ch_pos_lower = {k.lower(): v for k, v in ch_pos.items()}

template_chs = [ch.lower() for ch in template_45 if ch.lower() in ch_pos_lower]
your_chs     = [ch.lower() for ch in your_channels if ch.lower() in ch_pos_lower]

# ------------------------------------------------------------
# 4. Get 3D coordinates (x,y,z)
# ------------------------------------------------------------
pos_template_3d = np.array([ch_pos_lower[ch] for ch in template_chs])
pos_your_3d     = np.array([ch_pos_lower[ch] for ch in your_chs])

# Optional normalization to skull radius (unit sphere)
pos_template_3d /= np.max(np.linalg.norm(pos_template_3d, axis=1))
pos_your_3d     /= np.max(np.linalg.norm(pos_your_3d, axis=1))

# ------------------------------------------------------------
# 5. Compute 3D Euclidean distance matrix
# ------------------------------------------------------------
D = cdist(pos_template_3d, pos_your_3d, metric='euclidean')

print(f"\n✅ 3D Distance matrix computed!")
print(f"   Shape: {D.shape}")
print(f"   Template channels: {len(template_chs)}")
print(f"   Your headset channels: {len(your_chs)}\n")

# Example: show distances from template 'Fz'
if 'fz' in template_chs:
    idx_fz = template_chs.index('fz')
    print(f"Distances from template 'Fz' to your headset channels:")
    for ch, d in zip(your_chs, D[idx_fz]):
        print(f"  {ch.upper():<4s}: {d:.4f}")

# ------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------
np.save("distances_3D.npy", D)
np.save("template_channels.npy", np.array(template_chs))
np.save("your_channels.npy", np.array(your_chs))
print("\n💾 Saved: distances_3D.npy + channel names.")

# ------------------------------------------------------------
# 7. Visualize matrix as heatmap
# ------------------------------------------------------------
plt.figure(figsize=(10, 8))
im = plt.imshow(D, cmap='plasma', aspect='auto')
plt.colorbar(im, label="Distancia Euclídea 3D")
plt.xlabel("Tus canales (BrainAccess Cap)")
plt.ylabel("Canales del template (45)")
plt.xticks(ticks=range(len(your_chs)), labels=[ch.upper() for ch in your_chs], rotation=45)
plt.yticks(ticks=range(len(template_chs)), labels=[ch.upper() for ch in template_chs])
plt.title("Matriz de distancias 3D – MIRepNet Channel Template")
plt.tight_layout()
plt.show()

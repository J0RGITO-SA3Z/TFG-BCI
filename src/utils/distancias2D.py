"""
Compute 2D Euclidean distances between your EEG headset channels
and the 45-channel MIRepNet template using the standard 10-20 montage (MNE),
and visualize them as a heatmap.
Author: Jorge + ChatGPT (2025)
"""

import json
import os
import numpy as np
from scipy.spatial.distance import cdist
import mne
import matplotlib
matplotlib.use('TkAgg')  # <-- importante para evitar error en PyCharm
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
# 3. Load 10–20 montage (standard electrode coordinates)
# ------------------------------------------------------------
montage = mne.channels.make_standard_montage('standard_1020')
ch_pos = montage.get_positions()['ch_pos']

# Convert all keys to lowercase to match MNE format
ch_pos_lower = {k.lower(): v for k, v in ch_pos.items()}

# Filter channels that exist in the montage
template_chs = [ch.lower() for ch in template_45 if ch.lower() in ch_pos_lower]
your_chs     = [ch.lower() for ch in your_channels if ch.lower() in ch_pos_lower]

# ------------------------------------------------------------
# 4. Get 2D coordinates (drop z, project to plane)
# ------------------------------------------------------------
pos_template_3d = np.array([ch_pos_lower[ch] for ch in template_chs])
pos_your_3d     = np.array([ch_pos_lower[ch] for ch in your_chs])

pos_template_2d = pos_template_3d[:, :2]
pos_your_2d     = pos_your_3d[:, :2]

# Normalize to unit circle (optional)
pos_template_2d /= np.max(np.linalg.norm(pos_template_2d, axis=1))
pos_your_2d     /= np.max(np.linalg.norm(pos_your_2d, axis=1))

# ------------------------------------------------------------
# 5. Compute 2D Euclidean distance matrix
# ------------------------------------------------------------
D = cdist(pos_template_2d, pos_your_2d, metric='euclidean')

print(f"\n✅ Distance matrix (2D) computed!")
print(f"   Shape: {D.shape}")
print(f"   Template channels: {len(template_chs)}")
print(f"   Your headset channels: {len(your_chs)}\n")

# ------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------
np.save("distances_2D.npy", D)
np.save("template_channels.npy", np.array(template_chs))
np.save("your_channels.npy", np.array(your_chs))
print("💾 Saved: distances_2D.npy + channel names.\n")

# ------------------------------------------------------------
# 7. Visualize the 2D distance matrix
# ------------------------------------------------------------
plt.figure(figsize=(10, 8))
im = plt.imshow(D, cmap='viridis', aspect='auto')
plt.colorbar(im, label="Distancia Euclídea 2D")
plt.xlabel("Tus canales (BrainAccess Cap)")
plt.ylabel("Canales del template (45)")
plt.xticks(ticks=range(len(your_chs)), labels=[ch.upper() for ch in your_chs], rotation=45)
plt.yticks(ticks=range(len(template_chs)), labels=[ch.upper() for ch in template_chs])
plt.title("Matriz de Distancias 2D – MIRepNet Channel Template")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. Optional: print sample distances (for inspection)
# ------------------------------------------------------------
print("✅ Matriz de distancias (2D):")
for i, t in enumerate(template_chs[:10]):  # mostrar solo los primeros 10
    fila = "  ".join(f"{d:.3f}" for d in D[i])
    print(f"{t.upper():<4s} | {fila}")

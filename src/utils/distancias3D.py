"""
Compute 3D Euclidean distances between your EEG headset channels
and the 45-channel MIRepNet template using the standard 10-20 montage (MNE).
Author: Jorge + ChatGPT (2025)
"""

import numpy as np
from scipy.spatial.distance import cdist
import mne
import matplotlib
matplotlib.use('TkAgg')  # o 'Qt5Agg' si tienes PyQt5 instalado

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Your headset (BrainAccess Cap)
# ------------------------------------------------------------
your_channels = [
    "F4","FCZ","FZ","FC3","F3","CZ","FC4",
    "C4","CP4","P4","C3","CP3","PZ","CPZ","P3"
]

# ------------------------------------------------------------
# 2. MIRepNet 45-channel template (from paper)
# ------------------------------------------------------------
template_45 = [
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'
]

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

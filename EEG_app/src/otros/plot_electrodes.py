"""
Visualización de la disposición de electrodos EEG (BrainAccess HALO 16ch).

Colores:
  - Verde   : electrodos EEG activos de medición
  - Azul    : electrodo de referencia (REF)
  - Amarillo: electrodo bias
  - Rojo    : referencia de bias (BIAS REF)
  - Gris    : electrodos presentes en el casco pero no utilizados
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import mne

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuración de electrodos ───────────────────────────────────────────────

canales_activos   = ["C4", "F4", "FC2", "CP2", "Fz", "Cz",
                     "FC1", "F3", "P4", "CP6", "CP1", "C3", "CP5", "P3"]
electrodo_bias     = "Fp1"
electrodo_ref      = "Fp2"
electrodo_bias_ref = "Pz"

COLOR_ACTIVO   = "#4CAF50"
COLOR_REF      = "#2196F3"
COLOR_BIAS     = "#FFC107"
COLOR_BIAS_REF = "#F44336"
COLOR_INACTIVO = "#E0E0E0"


# ── Posiciones 2D a partir del montaje estándar MNE ──────────────────────────

def obtener_posiciones_electrodos(nombres: list[str]) -> dict[str, tuple[float, float]]:
    """Devuelve {nombre: (x, y)} proyectando las coords 3D del montaje 10-20."""
    montaje = mne.channels.make_standard_montage("standard_1020")
    posiciones_3d = montaje.get_positions()["ch_pos"]
    map_lower = {n.lower(): n for n in montaje.ch_names}

    posiciones_2d = {}
    for nombre in nombres:
        clave = map_lower.get(nombre.lower())
        if clave is None:
            print(f"[AVISO] '{nombre}' no encontrado en el montaje, se omite.")
            continue
        x, y, _ = posiciones_3d[clave]
        posiciones_2d[nombre] = (x, y)

    return posiciones_2d


def obtener_color(nombre: str) -> str:
    if nombre in canales_activos:
        return COLOR_ACTIVO
    if nombre == electrodo_bias:
        return COLOR_BIAS
    if nombre == electrodo_ref:
        return COLOR_REF
    if nombre == electrodo_bias_ref:
        return COLOR_BIAS_REF
    return COLOR_INACTIVO


# ── Figura principal ──────────────────────────────────────────────────────────

def visualizar_electrodos():
    todos = canales_activos + [electrodo_bias, electrodo_ref, electrodo_bias_ref]
    posiciones = obtener_posiciones_electrodos(todos)

    if not posiciones:
        print("No se encontraron posiciones válidas.")
        return

    _, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    # Contorno de la cabeza
    radio_cabeza = 0.11
    cabeza = plt.Circle((0, 0), radio_cabeza, fill=False,
                         linewidth=2, edgecolor="black")
    ax.add_patch(cabeza)

    # Nariz
    nariz_x = [-0.01, 0, 0.01]
    nariz_y = [radio_cabeza, radio_cabeza + 0.015, radio_cabeza]
    ax.plot(nariz_x, nariz_y, color="black", linewidth=2)

    # Orejas
    oreja_izq = mpatches.Arc((-radio_cabeza, 0), 0.02, 0.05, angle=0,
                              theta1=90, theta2=270, linewidth=2)
    oreja_der = mpatches.Arc((radio_cabeza, 0), 0.02, 0.05, angle=0,
                              theta1=-90, theta2=90, linewidth=2)
    ax.add_patch(oreja_izq)
    ax.add_patch(oreja_der)

    # Electrodos
    radio_electrodo = 0.008
    for nombre, (x, y) in posiciones.items():
        color = obtener_color(nombre)
        circulo = plt.Circle((x, y), radio_electrodo, color=color,
                              ec="black", linewidth=1.2, zorder=5)
        ax.add_patch(circulo)

        ax.text(x, y, nombre,
                ha="center", va="center", fontsize=6, fontweight="bold", zorder=6)

    # Leyenda
    leyenda = [
        Line2D([0], [0], marker="o", color="w", label="Activos",
               markerfacecolor=COLOR_ACTIVO, markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label=f"Bias ({electrodo_bias})",
               markerfacecolor=COLOR_BIAS, markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label=f"Referencia ({electrodo_ref})",
               markerfacecolor=COLOR_REF, markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label=f"Ref. de Bias ({electrodo_bias_ref})",
               markerfacecolor=COLOR_BIAS_REF, markeredgecolor="black", markersize=9),
    ]
    ax.legend(handles=leyenda, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9,
              frameon=True, fancybox=True)

    margen = 0.04
    ax.set_xlim(-radio_cabeza - margen, radio_cabeza + margen)
    ax.set_ylim(-radio_cabeza - margen, radio_cabeza + margen)
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(os.path.join(_DIR, "electrodos_layout.png"), dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    visualizar_electrodos()

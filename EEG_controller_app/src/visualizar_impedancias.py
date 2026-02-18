"""
Visualización gráfica de impedancias EEG.

Muestra cada electrodo como un círculo coloreado según su impedancia:
  - Verde  : impedancia buena   (< umbral_bueno kΩ)
  - Amarillo: impedancia aceptable (entre umbral_bueno y umbral_malo kΩ)
  - Rojo   : impedancia mala    (> umbral_malo kΩ)

Las posiciones 2-D de los electrodos se obtienen del montaje estándar
10-20 de MNE.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import mne


# ─── Umbrales de impedancia (kΩ) ────────────────────────────────────────────
UMBRAL_BUENO = 10   # < 10 kΩ → verde
UMBRAL_MALO  = 50   # > 50 kΩ → rojo  ;  entre 10-50 → amarillo


def obtener_color(impedancia: float) -> str:
    """Devuelve el color según el valor de impedancia."""
    if impedancia == 0:
        return "red"  
    elif impedancia < UMBRAL_BUENO:
        return "green"
    elif impedancia < UMBRAL_MALO:
        return "#FFD700"  # amarillo dorado
    else:
        return "#DA8B23"


def obtener_posiciones_electrodos(nombres: list[str]) -> dict[str, tuple[float, float]]:
    """
    Obtiene las coordenadas 2-D de cada electrodo a partir del montaje
    estándar 10-20 de MNE.

    Devuelve un diccionario {nombre_electrodo: (x, y)}.
    """
    montaje = mne.channels.make_standard_montage("standard_1020")

    # Nombres en el montaje están en formato capitalizado (e.g. 'Fz', 'Cz')
    nombres_montaje = montaje.ch_names
    posiciones_3d = montaje.get_positions()["ch_pos"]

    # Mapa nombre_lower → nombre_original del montaje para búsqueda flexible
    map_lower = {n.lower(): n for n in nombres_montaje}

    posiciones_2d = {}
    for nombre in nombres:
        clave = map_lower.get(nombre.lower())
        if clave is None:
            print(f"[AVISO] Electrodo '{nombre}' no encontrado en el montaje 10-20, se omite.")
            continue
        pos3d = posiciones_3d[clave]
        # Proyección simple: tomar x, y (ignorar z)
        posiciones_2d[nombre] = (pos3d[0], pos3d[1])

    return posiciones_2d


def visualizar_impedancias(
    electrodos: dict[str, float],
    titulo: str = "Test de Impedancias EEG",
    umbral_bueno: float | None = None,
    umbral_malo: float | None = None,
):
    """
    Genera un gráfico con la posición de cada electrodo coloreado según
    su impedancia.

    Parámetros
    ----------
    electrodos : dict[str, float]
        Diccionario {nombre_electrodo: impedancia_kOhm}.
    titulo : str
        Título del gráfico.
    umbral_bueno : float, opcional
        Sobreescribe el umbral «verde» (kΩ). Por defecto UMBRAL_BUENO.
    umbral_malo : float, opcional
        Sobreescribe el umbral «rojo» (kΩ). Por defecto UMBRAL_MALO.
    """
    global UMBRAL_BUENO, UMBRAL_MALO
    if umbral_bueno is not None:
        UMBRAL_BUENO = umbral_bueno
    if umbral_malo is not None:
        UMBRAL_MALO = umbral_malo

    nombres = list(electrodos.keys())
    posiciones = obtener_posiciones_electrodos(nombres)

    if not posiciones:
        print("No se encontraron posiciones válidas para los electrodos dados.")
        return

    # ── Dibujar contorno de la cabeza ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_title(titulo, fontsize=14, fontweight="bold", pad=20)

    # Círculo de la cabeza
    radio_cabeza = 0.11  # ~metros en coordenadas MNE
    cabeza = plt.Circle((0, 0), radio_cabeza, fill=False,
                         linewidth=2, edgecolor="black")
    ax.add_patch(cabeza)

    # Nariz (triángulo indicador de orientación)
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

    radio_electrodo = 0.008
    for nombre in nombres:
        if nombre not in posiciones:
            continue

        x, y = posiciones[nombre]
        imp = electrodos[nombre]
        color = obtener_color(imp)

        circulo = plt.Circle((x, y), radio_electrodo, color=color,
                              ec="black", linewidth=1.2, zorder=5)
        ax.add_patch(circulo)

        # Etiqueta del electrodo
        ax.text(x, y + radio_electrodo + 0.005, nombre,
                ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Valor de impedancia dentro del círculo
        ax.text(x, y, f"{imp:.0f}", ha="center", va="center",
                fontsize=6, color="black", zorder=6)

    leyenda_items = [
        mpatches.Patch(color="red",   label=f"desconectado (0 kΩ)"),
        mpatches.Patch(color="green",   label=f"Buena  (< {UMBRAL_BUENO} kΩ)"),
        mpatches.Patch(color="#FFD700",  label=f"Aceptable ({UMBRAL_BUENO}–{UMBRAL_MALO} kΩ)"),
        mpatches.Patch(color="red",     label=f"Mala   (> {UMBRAL_MALO} kΩ)"),
    ]
    ax.legend(handles=leyenda_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9,
              frameon=True, fancybox=True)

    margen = 0.04
    ax.set_xlim(-radio_cabeza - margen, radio_cabeza + margen)
    ax.set_ylim(-radio_cabeza - margen, radio_cabeza + margen + 0.02)
    ax.axis("off")

    plt.tight_layout()
    plt.show()


# ─── Ejemplo de uso ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Datos de ejemplo: nombre del electrodo → impedancia en kΩ
    datos_ejemplo = {
        "C4":  5.2,
        "F4":  8.1,
        "FC2": 12.4,
        "CP2": 3.7,
        "Fz":  45.0,
        "Cz":  6.3,
        "FC1": 22.8,
        "F3":  55.0,
        "P4":  9.9,
        "CP6": 15.3,
        "CP1": 4.1,
        "C3":  7.6,
        "Pz":  11.0,
        "CP5": 60.2,
        "P3":  2.5,
    }

    visualizar_impedancias(datos_ejemplo)

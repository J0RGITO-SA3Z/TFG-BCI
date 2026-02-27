"""
Utilidades de MiRepNet expuestas como paquete importable.

Exporta las variables y funciones de channel_list que se necesitan
externamente (posiciones de canales, listas de canales, etc.)
sin arrastrar las dependencias internas de utils.py.
"""

from .channel_list import channel_positions, use_channels_names, all_channels_names

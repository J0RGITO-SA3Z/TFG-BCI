from dataclasses import dataclass
import mne

@dataclass
class ChannelConfig:
    index: int
    name: str
    enabled: bool
    is_bias: bool
    electrode: str

## Verifica que el nombre del electrodo ingresado es valido según los nombres estándar de MNE. 
## Si el problema es de mayúsuclas o espacios, lo corrige automáticamente. Si el nombre no es reconocido, devuelve None.
def validar_nombre_electrodo(nombre):
    montage = mne.channels.make_standard_montage("standard_1005")
    nombres_mne = montage.ch_names

    nombre = nombre.strip().upper()
    mapa = {ch.upper(): ch for ch in nombres_mne}

    return mapa.get(nombre, None)
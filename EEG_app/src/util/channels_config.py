from dataclasses import dataclass
import json
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

# Funcion que carga la configuración de los canales de un archivo json. 
# Si el archivo contiene nombres de electrodos no reconocidos por MNE, devuelve None para indicar error. 
# En caso contrario, devuelve la lista de ChannelConfig cargada.
def load_channels_conf(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        electrodes = cfg["electrodes"]
    channels = []
    for ch in electrodes:
        channel = ChannelConfig(
            index=ch["index"],
            name= ch.get("electrode", f"CH{ch['index']+1}"),
            enabled=ch["active"],
            is_bias=ch["bias"],
            electrode= validar_nombre_electrodo(ch["name"]) 
        )
        channels.append(channel)

        if channel.electrode is None:
            return None 
    return channels
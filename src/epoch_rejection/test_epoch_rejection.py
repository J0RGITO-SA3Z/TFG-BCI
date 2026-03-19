import os
import sys
import numpy as np
import mne
import time
import matplotlib.pyplot as plt  # noqa

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ─────── Data Providers ─────────────────────────────────────────────────────
from DataProvider.FifDataProvider import FifDataProvider

# ─────── FASTER (Fully Automated Statistical Thresholding EEG Rejection) ─────────────────────────────────────────────────────
from mne_faster import ( # -> pip install mne_faster
    find_bad_channels,
    find_bad_channels_in_epochs,
    find_bad_components,
    find_bad_epochs,
)

# ─────── AutoReject (Automated artifact rejection for MEG and EEG data) ─────────────────────────────────────────────────────
from autoreject import AutoReject, get_rejection_threshold, compute_thresholds, set_matplotlib_defaults # -> pip install autoreject

# ─────── Imports pipeline ─────────────────────────────────────────────────────
from raw_processing.RawProcessorPipeline import RawProcessorPipeline
from raw_processing.BandpassFilter import BandpassFilter
from raw_processing.NotchFilter import NotchFilter
from raw_processing.Resampler import Resampler
from raw_processing.CARReference import CARReference
from raw_processing.ICAProcessor import ICAProcessor
from raw_processing.AnnotationRenamer import AnnotationRenamer

LABEL_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
    "DESCANSO":  "rest",
}

def _raw_to_epochs(raw, tmin=0.0, tmax=4.0, anotationsNames=["left_hand", "right_hand", "feet"]):
    """
    Epoquiza un Raw ya preprocesado por el pipeline.
    Las anotaciones ya están renombradas (left_hand, right_hand, feet)
    y el Raw ya tiene 45 canales gracias a SpatialInterpolator.
    """
    events, event_id = mne.events_from_annotations(raw)
    event_id_filtrado = {k: v for k, v in event_id.items() if k in anotationsNames}
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_filtrado,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True,
    )

    return epochs


def fif_to_epochs(path, anotationsNames=["left_hand", "right_hand", "feet"]):
    ''' 
    Lee un archivo .fif, lo preprocesa con el pipeline y lo epoquiza. 
    Devuelve un objeto mne.Epochs listo para aplicar FASTER.
    '''
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    raw = raw.set_eeg_reference('average')
    raw = raw.pick_types(meg=False, eeg=True, eog=False)
    #raw.filter(0.3, None, method="fir") # filtro de alta frecuencia para mejorar la detección de artefactos por parte de FASTER
    _raw_pipeline = RawProcessorPipeline([
                # NotchFilter(50.0),
                BandpassFilter(8.0, 30.0), # filtro de banda para mejorar la detección de artefactos por parte de AutoReject
                AnnotationRenamer(LABEL_MAP),
                # CARReference(),
                # Resampler(250),
                # ICAProcessor(),
            ])
    
    raw = _raw_pipeline.process(raw)
    epoch = _raw_to_epochs(raw, anotationsNames = anotationsNames)

    return epoch


def plot_epochs(epochs):
    ''' Función auxiliar para visualizar los epochs antes de aplicar FASTER. '''
    epoch_cop = epochs.copy()
    epoch_cop.set_eeg_reference("average")
    evoked = epoch_cop.average()
    evoked.plot()


# FASTER test functions ─────────────────────────────────────────────────────

def test_faster_thresholds(epochs):
    ''' 
    Prueba diferentes umbrales para FASTER y muestra los canales y epochs marcados como malos.
    Esto es útil para entender cómo afecta el umbral a la detección de artefactos. 
    Se puede comparar con los resultados obtenidos por nuestro pipeline para elegir un umbral adecuado.
    '''
    for th in [2.5, 3.0, 3.5, 4.0]:
        bad_chs = find_bad_channels(epochs, thres=th, eeg_ref_corr=False)
        bad_eps = find_bad_epochs(epochs, thres=th)
        print(f"Threshold = {th}")
        print("  Bad channels:", bad_chs)
        print("  Num bad epochs:", len(bad_eps))
  
def test_faster(epoch):

    thres_in = input("Escribe el umbral para FASTER (default: 3.0): ").strip()
    thres = float(thres_in) if thres_in else 3.0
    # Compute evoked before cleaning, using an average EEG reference
    epoch_before = epoch.copy()

    # Aplicamos FASTER a los epochs procesados por nuestro pipeline
    # No se puede palicar FASTER directamente porque no tenemos canales eog 
    # cleaned = run_faster(single_epoch, thres=3.0, copy=True)

    # Step 1: mark bad channels
    # montage = mne.channels.make_standard_montage("standard_1020")
    # epoch.set_montage(montage, on_missing="ignore")
    epoch.info["bads"] = find_bad_channels(epoch, thres=thres, eeg_ref_corr=False)
    
    # Step 2: bad epochs según FASTER
    bad_epoch_idx = find_bad_epochs(epoch,thres=thres)  # devuelve lista de índices

    # Step 3: bad channels in epochs
    bad_ch_in_epochs = find_bad_channels_in_epochs(epoch, thres=thres)

    print(f"FASTER bad channels : {epoch.info['bads']}")
    print(f"FASTER bad epochs   : {bad_epoch_idx}")
    print(f"Nº de epochs antes  : {len(epoch)}")

    print(type(bad_ch_in_epochs))
    print(bad_ch_in_epochs)

    # Interpolar canales malos (opcional pero recomendado antes de drop epochs)
    if epoch.info["bads"]:
        epoch.interpolate_bads(reset_bads=True)

    # Descartar epochs malos
    epoch.drop(bad_epoch_idx, reason="FASTER")

    # Interpolar canales malos en epochs
    for i, bads in enumerate(bad_ch_in_epochs):
        if not bads:
            continue

        ep = epoch[i].copy()   # Epochs con 1 solo epoch
        ep.info["bads"] = bads
        ep.interpolate_bads(reset_bads=True)

        epoch._data[i] = ep.get_data()[0]

    print(f"Nº de epochs después: {len(epoch)}")

    plot_epochs(epoch_before)  # visualiza los epochs antes de FASTER
    plot_epochs(epoch)  # visualiza los epochs limpios después de FASTER

# AutoReject test function ─────────────────────────────────────────────────────

def test_autoreject_global(epoch):
    ''' 
    Opción A — umbral global automático (rápido, 1 línea)
    Aprende el umbral óptimo por validación cruzada y descarta los epochs malos.
    Solo sirve para decartar epochs enteros, no para interpolar canales malos dentro de un epoch.
    '''

    inicio = time.perf_counter()
    # Opción A — umbral global automático (rápido, 1 línea)
    # Aprende el umbral óptimo por validación cruzada
    reject = get_rejection_threshold(epoch)  
    # → {'eeg': 0.000087}  (en voltios, MNE usa V no µV)
    epochs_clean = epoch.drop_bad(reject=reject)

    fin = time.perf_counter()
    print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")
    # Hay que hacerlo de esta forma para obtener el log con mne
    # 1. Qué trials se descartaron y por qué canal
    #print(epochs_clean.drop_log)
    # → ('IGNORED',) para los buenos
    # → ('EEG 003',) para los descartados, con el canal culpable

    # 2. Índices de trials descartados
    bad_idx = [i for i, log in enumerate(epochs_clean.drop_log) if len(log) > 0]
    print(f"Trials descartados: {bad_idx}")

    # 3. Resumen visual
    epochs_clean.plot_drop_log()  # gráfico de barras por canal
    plot_epochs(epochs_clean)  # visualiza los epochs limpios después de AutoReject

def test_autoreject_slow(epoch):
    '''
    Completo (más potente pero más lento) 
    que aprende umbrales distintos por canal y por trial, e interpola los canales malos en vez de descartar el trial entero.
    '''

    # Opción B — AutoReject completo (más potente)
    # Aprende umbrales DISTINTOS por electrodo, e interpola
    # los canales malos en cada trial en vez de descartar el trial entero
    ar = AutoReject(n_interpolate=[1, 2, 4],consensus=[0.3],random_state=42)

    inicio = time.perf_counter()

    ar.fit(epoch)
    epochs_clean, reject_log = ar.transform(epoch, return_log=True)

    fin = time.perf_counter()
    print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")

    # Ver qué trials/canales fueron problemáticos
    reject_log.plot()        # mapa de calor: trials × canales
    scalings = dict(eeg=1)
    reject_log.plot_epochs(epoch, scalings=scalings)  # visualiza los trials rechazados
    plot_epochs(epochs_clean)  # visualiza los epochs limpios después de AutoReject

    print(f"Nº de epochs después: {len(epochs_clean)}")
    print("Bad epochs:", reject_log.bad_epochs)
    print("Nº bad epochs:", reject_log.bad_epochs.sum())

def show_thresholds(epochs):
    picks = mne.pick_types(epochs.info, eeg=True)
    threshes = compute_thresholds(epochs, picks=picks, method='bayesian_optimization',
                                  random_state=42, augment=False, verbose=True)

    # EEG: los umbrales vienen en Voltios → convertir a µV
    unit    = 'µV'
    scaling = 1e6  # V → µV

    valores = scaling * np.array(list(threshes.values()))
    print(f"Rango de umbrales: {valores.min():.1f} – {valores.max():.1f} µV")

    plt.figure(figsize=(6, 5))
    plt.hist(valores, bins=30, color='steelblue', alpha=0.6, edgecolor='white')
    plt.xlabel(f'Umbral ({unit})')
    plt.ylabel('Número de canales')
    plt.xlim((valores.min() * 0.8, valores.max() * 1.2))  # ajuste automático
    plt.tight_layout()
    plt.show()

'''
en epoch processor

ar = AutoReject(
    n_interpolate=[1, 2, 4],
    consensus=[0.6, 0.7, 0.8],
    cv=10,
    picks='eeg',
    verbose=True
)

ar.fit(epochs_calib)          # aprende thresholds y parámetros
------------------------------------------

cuándos e vaya a usar:
epoch_clean, log = ar.transform(epoch_rt, return_log=True)
'''

# Reimannian Potato test function ─────────────────────────────────────────────────────

def test_riemannian_potato(epoch):
    return


def main() -> None:
    anotationsNames=["left_hand"]
    fif_path = "EEG_controller_app/recordings/suj4_3_raw.fif"
    epoch = fif_to_epochs(fif_path, anotationsNames=anotationsNames)

    #test_faster_thresholds(epochs=epoch)
    #test_faster(epoch=epoch)

    #test_autoreject_global(epoch=epoch)
    test_autoreject_slow(epoch=epoch)
    show_thresholds(epochs=epoch)


if __name__ == "__main__":
    main()

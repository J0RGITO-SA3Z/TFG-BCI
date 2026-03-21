'''
Autoreject (Automated artifact rejection for MEG and EEG data) - https://autoreject.github.io/
'''
import mne
import numpy as np
from torch import List, Optional

from epoch_processing.EpochProcessor import EpochProcessor


# ─────── AutoReject (Automated artifact rejection for MEG and EEG data) ─────────────────────────────────────────────────────
from autoreject import AutoReject, get_rejection_threshold, compute_thresholds, set_matplotlib_defaults



class Autoreject(EpochProcessor):
    
    def __init__(self, reject=None, actual_channel_positions: Optional[List[str]] = None, n_interpolate = [1, 2, 4], consensus = [0.3], plot=False, sfreq=250):
        self.autoreject = reject
        self.n_interpolate = n_interpolate
        self.consensus = consensus
        self.plot = plot
        self.ch_names = [self.validar_nombre_electrodo(ch) for ch in actual_channel_positions] if actual_channel_positions else None
        self.sfreq = sfreq  # Frecuencia de muestreo por defecto (ajustar según tus datos)

        super().__init__()

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        '''
        AutoReject completo 
        que aprende umbrales distintos por canal y por trial, e interpola los canales malos en vez de descartar el trial entero.
        '''
        if self.autoreject is None:
            self.autoreject = AutoReject(n_interpolate=[1, 2, 4],consensus=[0.3],random_state=42)
        
        self.autoreject.fit(epochs)
        epochs_clean, reject_log = self.autoreject.transform(epochs, return_log=True)

        self.summary(reject_log, epochs, epochs_clean)

        return epochs_clean

    def process_np(self, X: np.ndarray, y: np.ndarray | None = None):
        raise NotImplementedError(
            "Autoreject solo funciona con objetos mne.Epochs. "
            "No puede ejecutarse sobre arrays numpy sin la metadata de MNE."
        )
    
    def process_np(self, X: np.ndarray, y: np.ndarray | None = None):   
        """
        X: (n_trials, n_channels, n_samples)
        y: (n_trials,) etiquetas — se filtran igual que los trials rechazados

        Retorna: X_clean (n_trials_clean, n_channels, n_samples), y_clean
        """
        n_trials, n_ch, n_samples = X.shape

        # Construir info MNE
        ch_names = self.ch_names
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=self.sfreq,
            ch_types=['eeg'] * n_ch
        )

        # Construir eventos sintéticos (necesarios para EpochsArray)
        events = np.column_stack([
            np.arange(n_trials) * n_samples,  # onset en muestras
            np.zeros(n_trials, dtype=int),
            np.ones(n_trials, dtype=int)       # event_id=1 genérico
        ])

        epochs = mne.EpochsArray(
            X, info=info, events=events,
            event_id={'stim': 1}, tmin=0.0, verbose=False
        )

        # Aplicar AutoReject (reutiliza process)
        epochs_clean = self.process(epochs)

        # Recuperar numpy
        X_clean = epochs_clean.get_data()  # (n_trials_clean, n_ch, n_samples)

        # Filtrar y con la misma máscara de trials rechazados
        y_clean = None
        if y is not None:
            # bad_epochs es un array booleano (n_trials,): True = descartado
            keep_mask = ~self.autoreject.get_reject_log(epochs).bad_epochs
            y_clean = y[keep_mask]

        return X_clean, y_clean

    # ------------------------------------------------------------------
    # Funciones auxiliares
    # ------------------------------------------------------------------
    def validar_nombre_electrodo(self,nombre):
        montage = mne.channels.make_standard_montage("standard_1005")
        nombres_mne = montage.ch_names

        nombre = nombre.strip().upper()
        mapa = {ch.upper(): ch for ch in nombres_mne}

        return mapa.get(nombre, None)
    
    def plot_epochs(epochs):
        ''' Función auxiliar para visualizar los epochs antes de aplicar FASTER. '''
        epoch_cop = epochs.copy()
        epoch_cop.set_eeg_reference("average")
        evoked = epoch_cop.average()
        evoked.plot()

    def summary(self, reject_log, epochs,epochs_clean):
        ''' Función auxiliar para mostrar un resumen de los resultados de AutoReject. '''
        if self.plot:
            # Ver qué trials/canales fueron problemáticos
            reject_log.plot()                                   # mapa de calor: trials × canales
            scalings = dict(eeg=1)
            reject_log.plot_epochs(epochs, scalings=scalings)   # visualiza los trials rechazados
            self.plot_epochs(epochs_clean)                  # visualiza los epochs limpios después de AutoReject

        print(f"Resumen del Epoch Rejection: \n")
        print(f"Nº de epochs después: {len(epochs_clean)}\n")
        print("Bad epochs:", reject_log.bad_epochs, "\n")
        print("Nº bad epochs:", reject_log.bad_epochs.sum(),"\n")

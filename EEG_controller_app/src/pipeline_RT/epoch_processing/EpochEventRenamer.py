"""
Renombrado de eventos/etiquetas de epochs.

Permite sustituir nombres o ids de eventos por otros nombres tanto para
``mne.Epochs`` como para arrays numpy ``(X, y)``.
"""

import numpy as np
import mne

from epoch_processing.EpochProcessor import EpochProcessor


class EpochEventRenamer(EpochProcessor):
	"""
	Renombra eventos de acuerdo con un mapeo ``origen -> destino``.

	Parameters
	----------
	rename_map : dict[str | int, str]
		Diccionario de mapeo. Las claves pueden ser nombre de evento o id
		numérico y el valor debe ser el nuevo nombre del evento.
	"""

	def __init__(self, rename_map: dict[str | int, str]) -> None:
		if not isinstance(rename_map, dict) or len(rename_map) == 0:
			raise ValueError("rename_map debe ser un diccionario no vacio")

		for old_key, new_name in rename_map.items():
			if not isinstance(old_key, (str, int)):
				raise TypeError(
					"Las claves de rename_map deben ser str o int "
					f"(recibido {type(old_key).__name__})"
				)
			if not isinstance(new_name, str):
				raise TypeError(
					"Los valores de rename_map deben ser str "
					f"(recibido {type(new_name).__name__})"
				)

		self.rename_map = rename_map

	# ------------------------------------------------------------------
	# Interfaz MNE
	# ------------------------------------------------------------------

	def process(self, epochs: mne.Epochs) -> mne.Epochs:
		current_event_id = dict(epochs.event_id)

		# Invertir para resolver claves numéricas: id -> nombre
		id_to_name = {eid: name for name, eid in current_event_id.items()}
		new_event_id: dict[str, int] = {}

		for old_name, old_id in current_event_id.items():
			# Prioridad: mapeo por nombre, luego mapeo por id.
			if old_name in self.rename_map:
				target_name = self.rename_map[old_name]
			elif old_id in self.rename_map:
				target_name = self.rename_map[old_id]
			else:
				target_name = old_name

			# Si dos clases acaban en el mismo nombre pero distinto id, es ambiguo.
			if target_name in new_event_id and new_event_id[target_name] != old_id:
				raise ValueError(
					"El renombrado produce nombres duplicados con ids distintos: "
					f"'{target_name}' -> {new_event_id[target_name]} y {old_id}"
				)

			new_event_id[target_name] = old_id

		data = epochs.get_data()
		events = epochs.events.copy()

		return mne.EpochsArray(
			data,
			info=epochs.info.copy(),
			events=events,
			event_id=new_event_id,
			tmin=epochs.tmin,
			verbose=False,
		)

	# ------------------------------------------------------------------
	# Interfaz numpy
	# ------------------------------------------------------------------

	def process_np(
		self, X: np.ndarray, y: np.ndarray | None = None
	) -> tuple[np.ndarray, np.ndarray | None]:
		if y is None:
			return X, y

		y_out = np.array(y, copy=True)
		for old_label, new_label in self.rename_map.items():
			y_out[y_out == old_label] = new_label

		return X, y_out

	def __repr__(self) -> str:
		return f"EpochEventRenamer(rename_map={self.rename_map})"

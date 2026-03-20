import numpy as np
import mne
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Ellipse

from pyriemann.clustering import Potato
from pyriemann.estimation import Covariances
from pyriemann.utils.base import nearest_sym_pos_def


@dataclass
class RiemannianRejectLog:
    """
    Log estilo autoreject.

    channel_status:
        0 = good
        1 = interpolated
        2 = bad
    """
    bad_epochs: np.ndarray               # (n_epochs,) bool
    labels: np.ndarray                  # (n_epochs,) object/string
    bad_channels: np.ndarray            # (n_epochs, n_channels) bool
    potato_clean: np.ndarray            # (n_epochs,) bool
    n_bad_channels: np.ndarray          # (n_epochs,) int
    ch_names: List[str]
    channel_status: np.ndarray          # (n_epochs, n_channels) int
    robust_z: Optional[np.ndarray] = None

    def plot(
        self,
        title: str = "Reject log",
        figsize=(7.5, 6.5),
        show_legend: bool = True,
        fontsize: int = 12,
    ):
        """
        Mapa de calor estilo autoreject:
        - verde: good
        - azul: interpolated
        - rojo: bad
        """
        mat = self.channel_status

        # colores muy parecidos al estilo visual de autoreject
        cmap = ListedColormap([
            "#86d885",  # good
            "#1111ff",  # interpolated
            "#ff1a1a",  # bad
        ])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            origin="upper",
        )

        ax.set_xlabel("Channels", fontsize=fontsize + 4, fontfamily="serif")
        ax.set_ylabel("Epochs", fontsize=fontsize + 4, fontfamily="serif")

        ax.set_xticks(np.arange(len(self.ch_names)))
        ax.set_xticklabels(
            self.ch_names,
            rotation=90,
            fontsize=max(fontsize - 2, 8),
            fontfamily="serif",
            va="center",
        )

        ytick_step = max(1, len(mat) // 8)
        yticks = np.arange(0, len(mat), ytick_step)
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            [str(y) for y in yticks],
            fontsize=fontsize + 1,
            fontfamily="serif",
        )

        # rejilla blanca fina como en autoreject
        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)

        if show_legend:
            legend_elements = [
                Patch(facecolor="#86d885", edgecolor="#86d885", label="good"),
                Patch(facecolor="#1111ff", edgecolor="#1111ff", label="interpolated"),
                Patch(facecolor="#ff1a1a", edgecolor="#ff1a1a", label="bad"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.22),
                ncol=3,
                frameon=True,
                fontsize=fontsize + 2,
            )

        if title:
            ax.set_title(title, fontsize=fontsize + 2)

        plt.tight_layout()
        plt.show()

    def plot_robust_z(
        self,
        title: str = "Robust z-score por canal y epoch",
        figsize=(10, 6),
        cmap: str = "viridis",
    ):
        if self.robust_z is None:
            raise ValueError("No hay robust_z guardado en este reject log.")

        plt.figure(figsize=figsize)
        plt.imshow(self.robust_z, aspect="auto", interpolation="nearest", cmap=cmap)
        plt.colorbar(label="robust z-score")
        plt.xticks(np.arange(len(self.ch_names)), self.ch_names, rotation=90)
        plt.xlabel("Channels")
        plt.ylabel("Epochs")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def summary(self) -> Dict[str, Any]:
        return {
            "n_epochs": int(len(self.bad_epochs)),
            "n_bad_epochs": int(np.sum(self.bad_epochs)),
            "n_good_epochs": int(np.sum(~self.bad_epochs)),
            "mean_bad_channels_per_epoch": float(np.mean(self.n_bad_channels)),
            "n_interpolated_cells": int(np.sum(self.channel_status == 1)),
            "n_bad_cells": int(np.sum(self.channel_status == 2)),
        }


class RiemannianEpochRejector:
    """
    Detector híbrido:
    - Potato: detector global del trial en espacio riemanniano de covarianzas
    - Detector local: detecta canales raros por trial
    - Si hay pocos canales malos -> interpola
    - Si hay demasiados o Potato falla -> descarta
    """

    def __init__(
        self,
        potato_threshold: float = 3.0,
        cov_estimator: str = "scm",
        cov_kwargs: Optional[dict] = None,
        channel_z_thresh: float = 4.0,
        max_bad_channels_interp: int = 2,
        max_bad_channels_drop: int = 4,
        local_feature: str = "ptp",   # "ptp" o "std"
        refit_potato_after_interpolation: bool = False,
        cov_regularization: float = 1e-6,
        enforce_spd: bool = True,
        reject_entire_epoch_row_red: bool = True,
    ):
        self.potato_threshold = potato_threshold
        self.cov_estimator = cov_estimator
        self.cov_kwargs = cov_kwargs or {}
        self.channel_z_thresh = channel_z_thresh
        self.max_bad_channels_interp = max_bad_channels_interp
        self.max_bad_channels_drop = max_bad_channels_drop
        self.local_feature = local_feature
        self.refit_potato_after_interpolation = refit_potato_after_interpolation
        self.cov_regularization = cov_regularization
        self.enforce_spd = enforce_spd
        self.reject_entire_epoch_row_red = reject_entire_epoch_row_red

        self.potato = Potato(threshold=potato_threshold)
        self.cov = Covariances(estimator=cov_estimator, **self.cov_kwargs)

        self._fitted = False
        self._ch_median = None
        self._ch_mad = None
        self.ch_names_ = None

        # cache para visualizaciones
        self._last_covs = None
        self._last_potato_clean = None
        self._last_bad_channel_matrix = None
        self._fit_covs = None

    # =========================================================
    # Utils
    # =========================================================
    def _compute_local_feature(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_epochs, n_channels, n_times)
        returns: (n_epochs, n_channels)
        """
        if self.local_feature == "ptp":
            return np.ptp(X, axis=2)
        elif self.local_feature == "std":
            return np.std(X, axis=2)
        else:
            raise ValueError("local_feature debe ser 'ptp' o 'std'")

    def _robust_zscore(self, feats: np.ndarray) -> np.ndarray:
        """
        feats: (n_epochs, n_channels)
        """
        return 0.6745 * (feats - self._ch_median[None, :]) / self._ch_mad[None, :]

    def _interpolate_single_epoch(
        self,
        epochs: mne.Epochs,
        epoch_idx: int,
        bad_ch_names: List[str]
    ) -> mne.Epochs:
        """
        Interpola SOLO un epoch, creando un Epochs temporal de tamaño 1.
        """
        ep = epochs[epoch_idx:epoch_idx + 1].copy()
        ep.info["bads"] = bad_ch_names
        ep.interpolate_bads(reset_bads=True, verbose=False)
        return ep

    def _regularize_covariances(self, covs: np.ndarray) -> np.ndarray:
        """
        covs: (n_epochs, n_channels, n_channels)
        """
        covs = covs.copy()
        n_epochs, n_channels, _ = covs.shape

        if self.cov_regularization > 0:
            eye = np.eye(n_channels)[None, :, :]
            covs = covs + self.cov_regularization * eye

        if self.enforce_spd:
            covs = nearest_sym_pos_def(covs, reg=self.cov_regularization)

        return covs

    # =========================================================
    # Fit
    # =========================================================
    def fit(self, epochs: mne.Epochs):
        """
        Ajuste offline:
        1) Potato sobre covarianzas de trials
        2) estadísticas robustas por canal para detector local
        """
        X = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_times)
        self.ch_names_ = list(epochs.ch_names)

        covs = self.cov.transform(X)
        covs = self._regularize_covariances(covs)
        self._fit_covs = covs.copy()

        self.potato.fit(covs)

        feats = self._compute_local_feature(X)
        self._ch_median = np.median(feats, axis=0)
        mad = np.median(np.abs(feats - self._ch_median[None, :]), axis=0)
        self._ch_mad = np.maximum(mad, 1e-12)

        self._fitted = True
        return self

    # =========================================================
    # Core logic per trial
    # =========================================================
    def _process_single_trial(
        self,
        epochs: mne.Epochs,
        epoch_idx: int,
        potato_clean: bool,
        bad_channel_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        bad_channel_mask: (n_channels,) bool
        """
        bad_ch_names = [epochs.ch_names[i] for i in np.where(bad_channel_mask)[0]]
        n_bad = len(bad_ch_names)

        # Caso 1: Potato dice limpio
        if potato_clean:
            if n_bad == 0:
                return {
                    "keep": True,
                    "interpolated": False,
                    "reason": "clean",
                    "bad_ch_names": bad_ch_names,
                    "epoch_obj": epochs[epoch_idx:epoch_idx + 1].copy(),
                }

            if n_bad <= self.max_bad_channels_interp:
                ep_interp = self._interpolate_single_epoch(epochs, epoch_idx, bad_ch_names)
                return {
                    "keep": True,
                    "interpolated": True,
                    "reason": "few_bad_channels_interpolated",
                    "bad_ch_names": bad_ch_names,
                    "epoch_obj": ep_interp,
                }

            if n_bad >= self.max_bad_channels_drop:
                return {
                    "keep": False,
                    "interpolated": False,
                    "reason": "too_many_bad_channels",
                    "bad_ch_names": bad_ch_names,
                    "epoch_obj": None,
                }

            # zona gris
            ep_interp = self._interpolate_single_epoch(epochs, epoch_idx, bad_ch_names)
            return {
                "keep": True,
                "interpolated": True,
                "reason": "moderate_bad_channels_interpolated",
                "bad_ch_names": bad_ch_names,
                "epoch_obj": ep_interp,
            }

        # Caso 2: Potato dice artefactado globalmente
        if n_bad <= self.max_bad_channels_interp and n_bad > 0:
            ep_interp = self._interpolate_single_epoch(epochs, epoch_idx, bad_ch_names)

            X2 = ep_interp.get_data(copy=True)
            cov2 = self.cov.transform(X2)
            cov2 = self._regularize_covariances(cov2)
            potato_clean_after = bool(self.potato.predict(cov2)[0])

            if potato_clean_after:
                return {
                    "keep": True,
                    "interpolated": True,
                    "reason": "rescued_by_interpolation",
                    "bad_ch_names": bad_ch_names,
                    "epoch_obj": ep_interp,
                }

        return {
            "keep": False,
            "interpolated": False,
            "reason": "global_artifact",
            "bad_ch_names": bad_ch_names,
            "epoch_obj": None,
        }

    # =========================================================
    # Transform all epochs
    # =========================================================
    def transform(self, epochs: mne.Epochs, return_log: bool = True):
        """
        Procesa TODOS los trials de un objeto Epochs.

        Returns
        -------
        epochs_out: mne.Epochs con los buenos (algunos interpolados)
        reject_log: RiemannianRejectLog
        """
        if not self._fitted:
            raise RuntimeError("Debes hacer fit() antes de usar transform().")

        X = epochs.get_data(copy=True)
        n_epochs, n_channels, _ = X.shape

        # ---- 1) Potato global por trial ----
        covs = self.cov.transform(X)
        covs = self._regularize_covariances(covs)
        potato_clean = self.potato.predict(covs).astype(bool)

        self._last_covs = covs.copy()
        self._last_potato_clean = potato_clean.copy()

        # ---- 2) Detección local por trial y canal ----
        feats = self._compute_local_feature(X)
        robust_z = self._robust_zscore(feats)
        bad_channel_matrix = np.abs(robust_z) > self.channel_z_thresh

        self._last_bad_channel_matrix = bad_channel_matrix.copy()

        # ---- 3) Procesamiento trial a trial ----
        kept_epochs = []
        labels = []
        bad_epochs = np.zeros(n_epochs, dtype=bool)
        n_bad_channels = bad_channel_matrix.sum(axis=1)
        channel_status = np.zeros((n_epochs, n_channels), dtype=int)  # 0=good, 1=interp, 2=bad

        for i in range(n_epochs):
            result = self._process_single_trial(
                epochs=epochs,
                epoch_idx=i,
                potato_clean=bool(potato_clean[i]),
                bad_channel_mask=bad_channel_matrix[i]
            )

            labels.append(result["reason"])
            bad_idx = np.where(bad_channel_matrix[i])[0]

            if result["keep"]:
                kept_epochs.append(result["epoch_obj"])

                # verdes por defecto; malos interpolados en azul
                if result["interpolated"] and len(bad_idx) > 0:
                    channel_status[i, bad_idx] = 1

            else:
                bad_epochs[i] = True

                if self.reject_entire_epoch_row_red:
                    channel_status[i, :] = 2
                else:
                    if len(bad_idx) > 0:
                        channel_status[i, bad_idx] = 2
                    else:
                        channel_status[i, :] = 2

        # ---- 4) Concatenar epochs buenos ----
        if len(kept_epochs) > 0:
            epochs_out = mne.concatenate_epochs(kept_epochs)
        else:
            epochs_out = None

        reject_log = None
        if return_log:
            reject_log = RiemannianRejectLog(
                bad_epochs=bad_epochs,
                labels=np.array(labels, dtype=object),
                bad_channels=bad_channel_matrix,
                potato_clean=potato_clean,
                n_bad_channels=n_bad_channels,
                ch_names=list(epochs.ch_names),
                channel_status=channel_status,
                robust_z=robust_z,
            )

        return epochs_out, reject_log

    def fit_transform(self, epochs: mne.Epochs, return_log: bool = True):
        self.fit(epochs)
        return self.transform(epochs, return_log=return_log)

    # =========================================================
    # Helpers de visualización
    # =========================================================
    def plot_channel_heatmap(
        self,
        reject_log: RiemannianRejectLog,
        title: str = "",
        figsize=(7.5, 6.5),
    ):
        """
        Heatmap lo más parecido posible al de autoreject:
        - good = verde
        - interpolated = azul
        - bad = rojo
        """
        reject_log.plot(title=title, figsize=figsize)

    def plot_epoch_summary(
        self,
        reject_log: RiemannianRejectLog,
        figsize=(4.5, 6),
    ):
        """
        Resumen por trial:
        0 = limpio
        1 = interpolado
        2 = descartado
        """
        summary_status = np.zeros((len(reject_log.labels), 1), dtype=int)

        for i in range(len(reject_log.labels)):
            if reject_log.bad_epochs[i]:
                summary_status[i, 0] = 2
            elif np.any(reject_log.channel_status[i] == 1):
                summary_status[i, 0] = 1
            else:
                summary_status[i, 0] = 0

        cmap = ListedColormap([
            "#86d885",
            "#1111ff",
            "#ff1a1a",
        ])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            summary_status,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            origin="upper",
        )

        ax.set_xticks([0])
        ax.set_xticklabels(["state"], fontfamily="serif")
        ax.set_ylabel("Epochs", fontsize=15, fontfamily="serif")
        ax.set_title("Epoch summary", fontsize=14)

        ax.set_yticks(np.arange(0, summary_status.shape[0], max(1, summary_status.shape[0] // 8)))

        ax.set_xticks(np.arange(-0.5, 1.5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, summary_status.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)

        legend_elements = [
            Patch(facecolor="#86d885", edgecolor="#86d885", label="good"),
            Patch(facecolor="#1111ff", edgecolor="#1111ff", label="interpolated"),
            Patch(facecolor="#ff1a1a", edgecolor="#ff1a1a", label="bad"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.22),
            ncol=3,
            frameon=True,
        )

        plt.tight_layout()
        plt.show()

    # =========================================================
    # Visualización opcional de la "potato"
    # =========================================================
    def plot_potato(
        self,
        epochs: Optional[mne.Epochs] = None,
        use_last_transform: bool = True,
        figsize=(8, 7),
        title: str = "Visualización de la Riemannian Potato",
        method: str = "pca",
        show_center: bool = True,
        show_boundary: bool = True,
    ):
        """
        Visualiza la 'patata' en 2D de forma aproximada.

        epochs:
            Si se pasa, calcula covarianzas de esos epochs.
            Si None, usa las últimas covarianzas de transform().
        """
        if not self._fitted:
            raise RuntimeError("Debes hacer fit() antes de usar plot_potato().")

        from sklearn.decomposition import PCA
        from sklearn.manifold import MDS

        # 1) Obtener covarianzas
        if epochs is not None:
            X = epochs.get_data(copy=True)
            covs = self.cov.transform(X)
            covs = self._regularize_covariances(covs)
            potato_clean = self.potato.predict(covs).astype(bool)
        elif use_last_transform and self._last_covs is not None:
            covs = self._last_covs.copy()
            potato_clean = self._last_potato_clean.copy()
        else:
            raise ValueError(
                "No hay covarianzas para visualizar. "
                "Pasa epochs=... o llama antes a transform()."
            )

        n_epochs, n_channels, _ = covs.shape

        # 2) Vectorizar covarianzas con triángulo superior
        iu = np.triu_indices(n_channels)
        feats = np.array([c[iu] for c in covs])

        # 3) Proyección 2D
        if method.lower() == "pca":
            projector = PCA(n_components=2)
            Z = projector.fit_transform(feats)
        elif method.lower() == "mds":
            projector = MDS(n_components=2, random_state=42)
            Z = projector.fit_transform(feats)
        else:
            raise ValueError("method debe ser 'pca' o 'mds'")

        good_Z = Z[potato_clean]
        bad_Z = Z[~potato_clean]

        if len(good_Z) == 0:
            raise RuntimeError("No hay trials aceptados por Potato para dibujar la potato.")

        center_2d = good_Z.mean(axis=0)

        fig, ax = plt.subplots(figsize=figsize)

        if len(good_Z) > 0:
            ax.scatter(
                good_Z[:, 0], good_Z[:, 1],
                c="#2ecc71", label="Aceptado por Potato",
                alpha=0.8, edgecolors="black", s=55
            )

        if len(bad_Z) > 0:
            ax.scatter(
                bad_Z[:, 0], bad_Z[:, 1],
                c="#e74c3c", label="Rechazado por Potato",
                alpha=0.85, edgecolors="black", s=55
            )

        if show_center:
            ax.scatter(
                center_2d[0], center_2d[1],
                c="gold", edgecolors="black", s=180,
                marker="X", label="Centro de la potato"
            )

        # contorno visual aproximado
        if show_boundary and len(good_Z) >= 3:
            cov2d = np.cov(good_Z.T)
            vals, vecs = np.linalg.eigh(cov2d)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width, height = 2.0 * 2 * np.sqrt(np.maximum(vals, 1e-12))

            ell = Ellipse(
                xy=center_2d,
                width=width,
                height=height,
                angle=angle,
                facecolor="none",
                edgecolor="#1f3b73",
                linewidth=2.5,
                linestyle="--",
                label="Contorno visual de la potato"
            )
            ax.add_patch(ell)

        ax.set_title(title)
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
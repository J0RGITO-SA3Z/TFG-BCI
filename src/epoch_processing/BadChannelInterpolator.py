import numpy as np

from collections.abc import Iterable
from epoch_processing import EpochProcessor
from epoch_processing.BadChannelDetectors.BadChannelDetector import BadChannelDetector

# Channel spatial interpolation
from epoch_processing.SpatialInterpolator import SpatialInterpolator

class BadChannelInterpolator(EpochProcessor):
    
    def __init__(self,detectors: Iterable[BadChannelDetector] | None = None,actual_channel_positions: Optional[List[str]] = None):
        self.detectors: list[BadChannelDetector] = list(detectors) if detectors else []
        self.bad_channel_list = []
        self.actual_channel_positions = actual_channel_positions

    def process_epoch(self, epoch):
        return epoch
    
    def process_np(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Process a batch numpy array `X` with shape (B, C, T).
        For each sample in the batch, run detectors and remove the
        reported channel indices from that sample individually.

        If all resulting samples have the same channel count the
        function returns a stacked ndarray `(B, C_new, T)`, otherwise
        it returns a list of 2D arrays (C_b, T) for each batch entry.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (B,C,T), got shape {X.shape}")

        B, C, T = X.shape
        new_X_list = []
        new_y_list = [] if y is not None else None

        for b in range(B):
            # Pass a single-sample batch to detectors to keep behaviour consistent
            xb_batch = X[b:b+1]  # shape (1, C, T)

            bad_indices = []
            for detector in self.detectors:
                out = detector.process(xb_batch)
                if out is None:
                    continue
                if isinstance(out, Iterable) and not isinstance(out, (str, bytes)):
                    for idx in out:
                        try:
                            i = int(idx)
                        except Exception:
                            continue
                        if 0 <= i < C:
                            bad_indices.append(i)
                else:
                    try:
                        i = int(out)
                    except Exception:
                        continue
                    if 0 <= i < C:
                        bad_indices.append(i)

            bad_indices = sorted(set(bad_indices))

            # Take X[b] as 2D (C, T)
            xb2 = X[b]
            if bad_indices:
                # If we have actual channel names, use SpatialInterpolator to
                # reconstruct the full original channel set via interpolation.
                if self.actual_channel_positions is None:
                    # fallback: delete channels if no names provided
                    xb2 = np.delete(xb2, bad_indices, axis=0)
                else:
                    if len(self.actual_channel_positions) != C:
                        raise ValueError(
                            f"Length of actual_channel_positions ({len(self.actual_channel_positions)}) does not match C ({C})"
                        )
                    # remaining channel names after removing bad indices
                    remaining_names = [ch for i, ch in enumerate(self.actual_channel_positions) if i not in bad_indices]

                    # temporary array without bad channels
                    xb_temp = np.delete(xb2, bad_indices, axis=0)

                    # interpolate back to original channel set
                    interpolator = SpatialInterpolator(
                        target_channels=self.actual_channel_positions,
                        actual_channel_positions=remaining_names,
                    )
                    try:
                        interp_X, _ = interpolator.process_np(xb_temp[np.newaxis, ...], None)
                        xb2 = interp_X[0]
                    except Exception:
                        # on failure, fallback to deletion
                        xb2 = xb_temp

            new_X_list.append(xb2)
            if new_y_list is not None:
                new_y_list.append(y[b])

        # If all samples have same channel count, stack into ndarray
        channel_counts = [arr.shape[0] for arr in new_X_list]
        if all(c == channel_counts[0] for c in channel_counts):
            X_new = np.stack(new_X_list, axis=0)  # shape (B, C_new, T)
            y_new = (np.array(new_y_list) if new_y_list is not None else None)
            return X_new, y_new

        # Otherwise return list of arrays and corresponding y list
        y_out = (new_y_list if new_y_list is not None else None)
        return new_X_list, y_out

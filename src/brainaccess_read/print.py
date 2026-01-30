import mne
import matplotlib.pyplot as plt
import numpy as np

raw = mne.io.read_raw_fif("20260129_215029_brainaccess_midi_15ch_raw.fif", preload=True, verbose=False)

raw.compute_psd(fmin=1, fmax=60, method="welch").plot()
plt.show()

raw = mne.io.read_raw_fif("20260129_215029_brainaccess_midi_15ch_raw.fif", preload=True, verbose=False)
raw.filter(1, 40, verbose=False)

data = raw.get_data(picks="eeg")   # 👈 YA está en µV
sfreq = raw.info["sfreq"]
t = np.arange(data.shape[1]) / sfreq

plt.figure(figsize=(14, 6))

for i, ch in enumerate(raw.ch_names):
    plt.plot(t, data[i], label=ch)

plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (µV)")
plt.title("EEG BrainAccess MIDI – µV vs tiempo")
plt.legend(ncol=3, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()
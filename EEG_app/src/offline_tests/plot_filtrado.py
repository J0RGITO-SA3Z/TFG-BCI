import mne
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RAW_PATH = "EEG_app/recordings/experimento_visual/suj1/suj1_1_raw.fif"
CANAL    = 0

raw = mne.io.read_raw_fif(RAW_PATH, preload=True, verbose=False)
ch_name = raw.ch_names[CANAL]

data, times = raw[CANAL, :]
data = data[0]
t    = times - times[0]

raw_filt = raw.copy()
raw_filt.filter(l_freq=5.0, h_freq=None, verbose=False)
data_filt, _ = raw_filt[CANAL, :]
data_filt = data_filt[0]

int_fmt = ticker.FuncFormatter(lambda x, _: f"{int(x)}")

fig1, ax1 = plt.subplots(figsize=(5, 4))
ax1.plot(t, data, color="#e05a5a", linewidth=0.6)
ax1.set_title("Señal cruda")
ax1.set_xlabel("Tiempo (s)")
ax1.set_ylabel("Amplitud (µV)")
ax1.yaxis.set_major_formatter(int_fmt)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figura_canal_crudo.png", dpi=150, bbox_inches="tight")
plt.show()

fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.plot(t, data_filt, color="#4a90d9", linewidth=0.6)
ax2.set_title("Señal filtrada")
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel("Amplitud (µV)")
ax2.yaxis.set_major_formatter(int_fmt)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figura_canal_filtrado.png", dpi=150, bbox_inches="tight")
plt.show()

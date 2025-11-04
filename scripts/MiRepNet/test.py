from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery
import numpy as np, os

SAVE_DIR = r"C:\Users\JORGE\OneDrive\Escritorio\Modelos BCI\Modelos\MIRepNet\data\PhysioNetMI"
os.makedirs(SAVE_DIR, exist_ok=True)

dataset = PhysionetMI()       # sin guiones raros ni bugs
paradigm = MotorImagery(n_classes=2)  # izquierda/derecha

X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2, 3])

print("✅ Datos:", X.shape)
print("🧠 Etiquetas:", np.unique(y))

np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y)

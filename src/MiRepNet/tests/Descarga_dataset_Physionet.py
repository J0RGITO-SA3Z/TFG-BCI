from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery
import numpy as np, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
SAVE_DIR = os.path.join(MIREPNET_DIR, "data", "PhysioNetMI")

os.makedirs(SAVE_DIR, exist_ok=True)

dataset = PhysionetMI()       # sin guiones raros ni bugs
paradigm = MotorImagery(n_classes=2)  # izquierda/derecha

X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2, 3])

print("✅ Datos:", X.shape)
print("🧠 Etiquetas:", np.unique(y))

np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y)

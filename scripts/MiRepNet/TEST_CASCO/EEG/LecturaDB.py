import sqlite3
import pandas as pd

# Ruta al archivo .db
db_path = r"C:\Users\JORGE\Documents\GitHub\TFG-BCI\Grabaciones casco\db\subj-1_ses-S001_task-_run-001_20251114_164813_eeg.db"

# Conexión
conn = sqlite3.connect(db_path)

# Ver las tablas que contiene
tablas = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tablas disponibles:")
print(tablas)

# Elegir una tabla y leerla
df = pd.read_sql_query("SELECT * FROM nombre_de_tu_tabla;", conn)

print(df.head())

conn.close()

import multiprocessing
import random
import time
import pandas as pd
import numpy as np
import pygame
from rich.console import Console

# Importaciones de BrainAccess según vuestro nuevo script
from brainaccess.utils.acquisition import EEG
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel

# Generar acciones random (según las acciones pasadas), Interfaz grafica como un objeto, 

# --- CONFIGURACIÓN ---
PORT = "COM6" 
SFREQ = 250
# Nombres de los 16 canales según vuestro Prueba.csv
CH_NAMES = ["P8", "O2", "P4", "C4", "F8", "F4", "Oz", "Cz", 
            "Fz", "Pz", "F3", "O1", "P7", "C3", "P3", "F7"]

# Marcadores
MARKER_REST = 0
MARKER_CROSS = 1
MARKER_CIRCLE = 2
MARKER_LEFT_ARROW = 3
MARKER_RIGHT_ARROW = 4
MARKER_DOWN_ARROW = 5


def generar_lista(acciones, total):
    n = len(acciones)
    if total % n != 0:
        raise ValueError("El total debe ser múltiplo del número de acciones")

    lista = acciones * (total // n)
    random.shuffle(lista)
    return lista

# --- PROCESO 1: PROTOCOLO VISUAL (Pygame) ---
def visual_protocol(queue_markers, queue_ready):
    pygame.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
    font = pygame.font.SysFont("Arial", 80)

    def draw_text(text, color):
        screen.fill((0, 0, 0))
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(info.current_w//2, info.current_h//2))
        screen.blit(surf, rect)
        pygame.display.flip()

    draw_text("Esperando conexión del casco...", (255, 255, 0))
    
    # Esperar señal de que el objeto EEG está listo y grabando
    status = queue_ready.get()
    if status != "READY": return
    actions = ["<<<", ">>>", "vvv"]
    time.sleep(2)
    trials = 1 # Ajustar según necesidad
    labels = generar_lista(actions,trials)
    try:
        for i in range(trials):
            # t=0s: Cruz
            queue_markers.put(MARKER_CROSS)
            draw_text("+", (255, 255, 255))
            time.sleep(2)

            # t=2s: Círculo
            queue_markers.put(MARKER_CIRCLE)
            # Dibujar círculo...
            time.sleep(1)

            # t=3s: Acción (Flecha)
            marker = [MARKER_LEFT_ARROW, MARKER_RIGHT_ARROW, MARKER_DOWN_ARROW][i % 3]
            label = labels[i % 3]
            queue_markers.put(marker)
            draw_text(label, (0, 255, 0))
            time.sleep(6)

            # t=9s: Descanso
            queue_markers.put(MARKER_REST)
            draw_text("Descanso", (200, 200, 200))
            time.sleep(3)

    finally:
        queue_markers.put("STOP")
        pygame.quit()

# --- PROCESO 2: GRABADOR (Siguiendo vuestro nuevo script) ---
def eeg_recorder(queue_markers, queue_ready, filename="Grabacion_Protocolo"):
    console = Console()
    
    # 1. MAPEADO CORRECTO: { Índice_Hardware: "Nombre_Electrodo" }
    electrodes = {i: CH_NAMES[i] for i in range(16)}

    with EEGManager() as mgr:
        eeg = EEG(mode="accumulate")
        try:
            eeg.setup(
                mgr=mgr,
                port=PORT,
                cap=electrodes,
                gain=8,
                bias=[15] 
            )
        except Exception as e:
            console.print(f"[red]Error en setup: {e}[/red]")
            queue_ready.put("ERROR")
            return

        # 2. Iniciar adquisición
        eeg.start_acquisition()
        queue_ready.put("READY")
        console.print("[green]Grabación iniciada y sincronizada.[/green]")

        running = True
        while running:
            try:
                while not queue_markers.empty():
                    msg = queue_markers.get_nowait()
                    if msg == "STOP":
                        running = False
                    else:
                        eeg.annotate(str(msg)) 
            except: pass
            time.sleep(0.01)

        # 3. Finalizar
        eeg.stop_acquisition()
        raw = eeg.get_mne() 
        eeg.close()

        # --- TRANSFORMACIÓN A CSV ---
        console.print("Generando CSV...")
        
        # ### CORRECCIÓN AQUÍ ###
        # raw.get_data() devuelve (21, n_muestras). 
        # Seleccionamos solo las primeras 16 filas (los electrodos)
        data = raw.get_data()[:16, :]  # <--- ESTA ES LA CLAVE
        
        times = raw.times     
        sfreq = raw.info['sfreq']
        
        # Reconstruir columna de marcadores
        marker_column = np.zeros(len(times))
        for ann in raw.annotations:
            start_idx = int(ann['onset'] * sfreq)
            val = int(ann['description'])
            marker_column[start_idx:] = val

        # Timestamp absoluto
        start_time = time.time() - times[-1]
        absolute_timestamps = start_time + times

        # DataFrame final
        # Ahora data.T tendrá 16 columnas y coincidirá con CH_NAMES
        df = pd.DataFrame(data.T, columns=CH_NAMES)
        df['Marker'] = marker_column.astype(int)
        df['Timestamp'] = absolute_timestamps

        df.to_csv(filename, index=False, sep=';')
        console.print(f"[bold green]✅ CSV guardado:[/bold green] {filename}.csv")

if __name__ == "__main__":
    q_markers = multiprocessing.Queue()
    q_ready = multiprocessing.Queue()

    p_vis = multiprocessing.Process(target=visual_protocol, args=(q_markers, q_ready))
    p_rec = multiprocessing.Process(target=eeg_recorder, args=(q_markers, q_ready))

    p_rec.start()
    p_vis.start()

    p_vis.join()
    p_rec.join()
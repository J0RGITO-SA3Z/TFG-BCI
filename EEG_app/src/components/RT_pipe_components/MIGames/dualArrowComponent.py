from __future__ import annotations

import pygame
import sys
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from typing import Optional

from components.RT_pipe_components.MIGames.MIGame import MIGame


class DualArrowComponent(MIGame):
    # =========================================================
    # CONFIGURACIÓN DE COLORES (MODIFICA AQUÍ MANUALMENTE)
    # =========================================================
    BG_COLOR = (0, 0, 0)                 # Fondo negro
    OUTLINE_COLOR = (20, 50, 120)        # Azul oscuro (contorno)
    FILL_COLOR = (70, 190, 255)          # Azul claro brillante (relleno)
    # =========================================================

    def __init__(self, pipe: Connection, stop_event: Event, left_pct=0.0, right_pct=0.0, nSteps=20):
        super().__init__(pipe, stop_event)
        self.left_pct = self._clamp(left_pct)
        self.right_pct = self._clamp(right_pct)
        self.nSteps = nSteps
        self.step = 100 / nSteps

        pygame.init()
        self.screen = pygame.display.set_mode((800, 400), pygame.RESIZABLE)
        pygame.display.set_caption("BCI Dual Arrow")
        self.clock = pygame.time.Clock()

    def _clamp(self, value):
        """Asegura que el valor esté siempre entre 0 y 100"""
        return max(0.0, min(100.0, float(value)))

    # --- MÉTODOS DE INTERACCIÓN (API DEL OBJETO) ---

    def set_left(self, pct):
        self.left_pct = self._clamp(pct)

    def set_right(self, pct):
        self.right_pct = self._clamp(pct)

    def set_both(self, left_pct, right_pct):
        self.left_pct = self._clamp(left_pct)
        self.right_pct = self._clamp(right_pct)

    def reset(self):
        self.left_pct = 0.0
        self.right_pct = 0.0

    def get_percentages(self):
        """Devuelve una tupla: (porcentaje_izquierdo, porcentaje_derecho)"""
        return self.left_pct, self.right_pct
    
    def add_right(self):
        if self.left_pct == 0:
            self.right_pct = self._clamp(self.step + self.right_pct)
        else:
            self.left_pct = self._clamp(self.left_pct - 2*self.step) # Cambiar aquí para que al "fallar" se resetee o se reste
        
        if self.right_pct == 100:
            OUTLINE_COLOR = (172, 193, 242)        # Azul muy clarito (contorno)
            return True
        else:
            OUTLINE_COLOR = (20, 50, 120)        # Azul oscuro (contorno)
            return False
        
    def add_left(self):
        if self.right_pct == 0:
            self.left_pct = self._clamp(self.step + self.left_pct)
        else:
            self.right_pct = self._clamp(self.right_pct - 2*self.step) # Cambiar aquí para que al "fallar" se resetee o se reste
        
        if self.left_pct == 100:
            OUTLINE_COLOR = (172, 193, 242)        # Azul muy clarito (contorno)
            return True
        else:
            OUTLINE_COLOR = (20, 50, 120)        # Azul oscuro (contorno)
            return False
        
    def sub_both(self):
        if self.right_pct > 0:
            self.right_pct = self._clamp(self.right_pct - self.step)
        elif self.left_pct > 0:
            self.left_pct = self._clamp(self.left_pct - self.step)

        if self.left_pct < 100 and self.right_pct < 100:
            OUTLINE_COLOR = (20, 50, 120)        # Azul oscuro (contorno)

    # --- LÓGICA DE RENDERIZADO ---

    def draw(self, surface):
        """
        Dibuja las flechas en la superficie proporcionada, escalándose
        proporcionalmente al tamaño de dicha superficie.
        """
        surface.fill(self.BG_COLOR) # Fondo
        
        w, h = surface.get_size() # Sacamos las dimensiones de la ventana para hacerlo proporcional
        center_x, center_y = w // 2, h // 2 # Sacamos las coordenadas de los centros

        # 1. Definir proporciones basadas en la ventana
        # El hueco entre las flechas será el 6% del ancho total
        gap = w * 0.06
        
        # El ancho máximo disponible para una flecha es casi la mitad de la ventana: la mitad de la ventana sin contar el hueco y dejando márgen para los bordes
        max_arrow_w = (w / 2) - (gap / 2) - (w * 0.05) # 5% de margen a cada borde
        max_arrow_h = h * 0.6 # Altura máxima: 60% de la ventana

        # Mantener proporción de la flecha (Aspect Ratio): Ancho = 1.5 * Alto
        arrow_w = min(max_arrow_w, max_arrow_h * 1.5)
        arrow_h = arrow_w / 1.5

        # Puntos base de una flecha apuntando a la DERECHA (normalizados de 0 a 1)
        # (x, y) donde x=0 es la base (centro de pantalla) y x=1 es la punta
        base_points = [
            (0.01, 0.3),   # Base superior
            (0.55, 0.3),   # Esquina interior superior
            (0.55, 0.1),   # Ala superior
            (0.99, 0.5),   # Punta
            (0.55, 0.9),   # Ala inferior
            (0.55, 0.7),   # Esquina interior inferior
            (0.01, 0.7)    # Base inferior
        ]

        # Grosor del contorno dinámico según el tamaño
        line_thickness = max(2, int(arrow_h * 0.03)) # El máximo entre 2 y el 3% de la altura de la flecha

        # AUXILIAR PARA ARREGLAR EL DESFASE DEL RELLENO
        apanyo = line_thickness/2

        # --- DIBUJAR FLECHA DERECHA ---
        # Polígono escalado a píxeles
        right_poly = [(int(x * (arrow_w - line_thickness/2)), int(y * arrow_h)) for x, y in base_points] # Para cada punto de la normalización lo escalamos a la altura y grosor reales
        
        # Superficie temporal transparente para la flecha derecha
        right_surf = pygame.Surface((int(arrow_w), int(arrow_h)), pygame.SRCALPHA) # La flag pygame.SRCALPHA marca que la superficie sea transparente
        
        if self.right_pct > 0:
            # Recortar el área de dibujo desde la izquierda (la base/centro) hacia la derecha
            clip_w = int(arrow_w * (self.right_pct / 100.0)) # Definimos un recorte del ancho desde el centro
            right_surf.set_clip(pygame.Rect(0, 0, clip_w, int(arrow_h))) # Aplicamos el recorte con un rectángulo del ancho definido
            pygame.draw.polygon(right_surf, self.FILL_COLOR, right_poly) # Dibujamos la flecha rellena de azul clarito, hasta donde corresponda
            right_surf.set_clip(None) # Quitar recorte
        
        # Dibujar contorno
        pygame.draw.polygon(right_surf, self.OUTLINE_COLOR, right_poly, line_thickness) # Dibujamos el contorno de la flecha para que se superponga al relleno
        
        # Colocar flecha derecha en la pantalla principal
        right_pos_x = center_x + (gap / 2)
        right_pos_y = center_y - (arrow_h / 2)
        surface.blit(right_surf, (right_pos_x, right_pos_y)) # Montamos el surface sobre la ventana completa

        # --- DIBUJAR FLECHA IZQUIERDA ---
        # Reflejar los puntos horizontalmente para que apunte a la izquierda
        # x=1 es la base (centro pantalla), x=0 es la punta
        #left_poly = [(int((1.0 - x) * arrow_w), int(y * arrow_h)) for x, y in base_points]
        left_poly = [(int((1.0 - x) * (arrow_w - line_thickness/2)), int(y * arrow_h)) for x, y in base_points]

        left_surf = pygame.Surface((int(arrow_w), int(arrow_h)), pygame.SRCALPHA)
        
        if self.left_pct > 0:
            # Recortar el área de dibujo desde la derecha (la base/centro) hacia la izquierda
            clip_w = int(arrow_w * (self.left_pct / 100.0)) # Definimos un recorte del ancho desde la izquierda
            clip_x = int(arrow_w - clip_w) # Definimos dónde empieza el rectángulo (lo movemos a la dcha para que el vacío quede a la izquierda)
            left_surf.set_clip(pygame.Rect(clip_x - apanyo, 0, clip_w, int(arrow_h))) # Aplicamos el recorte con un rectángulo del ancho definido
            pygame.draw.polygon(left_surf, self.FILL_COLOR, left_poly) # Dibujamos la flecha rellena de azul clarito, hasta donde corresponda
            left_surf.set_clip(None) # Quitar recorte
            
        pygame.draw.polygon(left_surf, self.OUTLINE_COLOR, left_poly, line_thickness) # Dibujamos el contorno de la flecha para que se superponga al relleno
        
        # Colocar flecha izquierda
        left_pos_x = center_x - (gap / 2) - arrow_w
        left_pos_y = center_y - (arrow_h / 2)
        surface.blit(left_surf, (left_pos_x, left_pos_y)) # Montamos el surface sobre la ventana completa

    def start(self, filename: Optional[str] = None) -> None:
        running = True
        while running and not self.stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            while self.pipe.poll(0):
                try:
                    prediction, _ = self.read_msg()
                    if prediction == "right_hand":
                        self.add_right()
                    elif prediction == "left_hand":
                        self.add_left()
                    elif prediction == "rest":
                        self.sub_both()
                except Exception:
                    pass

            self.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


# =========================================================
# BLOQUE DE PRUEBA Y DEMOSTRACIÓN
# Este bloque solo se ejecuta si corres este script directamente.
# Si lo importas desde otro script, esto será ignorado.
# =========================================================
if __name__ == "__main__":
    import multiprocessing
    _conn_recv, _conn_send = multiprocessing.Pipe(duplex=False)
    _stop = multiprocessing.Event()

    arrows = DualArrowComponent(_conn_recv, _stop)

    # Variables para animar de forma independiente por consola/teclas
    running = True

    print("--- CONTROLES DE PRUEBA ---")
    print("A / D : Bajar/Subir porcentaje IZQUIERDO")
    print("Flecha Izq/Der : Bajar/Subir porcentaje DERECHO")
    print("R : Resetear ambas a 0%")
    print("Ajusta el tamaño de la ventana para probar el escalado.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                l_pct, r_pct = arrows.get_percentages()
                llena = False
                
                # Controles flecha izquierda (A y D)
                if event.key == pygame.K_a:
                    #arrows.set_left(l_pct - 5)
                    arrows.add_left()
                elif event.key == pygame.K_d:
                    #arrows.set_left(l_pct + 5)
                    arrows.add_right()
                    
                # Controles flecha derecha (Flechas del teclado)
                elif event.key == pygame.K_LEFT:
                    arrows.set_right(r_pct - 5)
                elif event.key == pygame.K_RIGHT:
                    arrows.set_right(r_pct + 5)

                elif event.key == pygame.K_l:
                    llena = arrows.add_right()
                elif event.key == pygame.K_j:
                    llena = arrows.add_left()
                    
                # Reset
                elif event.key == pygame.K_r:
                    arrows.reset()

                if llena:
                    print("SE HA LLENADOOO")

        # Dibujar el componente en la pantalla principal
        arrows.draw(arrows.screen)

        pygame.display.flip()
        arrows.clock.tick(60)

    pygame.quit()
    sys.exit()
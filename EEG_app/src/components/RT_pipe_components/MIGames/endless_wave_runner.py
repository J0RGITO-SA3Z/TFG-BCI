from __future__ import annotations

import random
import sys
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from typing import Optional

import pygame

from components.RT_pipe_components.MIGames.MIGame import MIGame

PREDICTED_LEFT  = pygame.USEREVENT + 10
PREDICTED_RIGHT = pygame.USEREVENT + 11
PREDICTED_REST  = pygame.USEREVENT + 12

AIMBOT_MODE = True
SEVERIDAD_AIMBOT = 1


class EndlessWaveRunnerMIGame(MIGame):
    def __init__(self, pipe: Connection, stop_event: Event) -> None:
        super().__init__(pipe, stop_event)
        pygame.init()

        # =====================================================================
        # CONSTANTES DE CONFIGURACIÓN DEL JUEGO (Totalmente modificables)
        # =====================================================================

        # --- Configuración General ---
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.TICK_RATE = 60  # FPS, determina el tiempo entre tick y tick

        # --- Configuración de la Nave ---
        self.SHIP_FIXED_Y = 450  # Coordenada Y estática donde se queda la nave
        self.SHIP_X_SPEED = 1.0  # Cuánto avanza la nave en X por cada tick
        self.SHIP_SIZE = 15      # Tamaño (radio) de la hitbox de la nave

        # --- Configuración del Entorno y Scroll ---
        self.INITIAL_SCROLL_SPEED = 1.0  # Velocidad a la que el mapa y obstáculos bajan
        self.ENABLE_OBSTACLES = True     # Activa/Desactiva los pinchos laterales

        # --- Configuración de Dificultad Base ---
        self.INITIAL_GAP_SIZE = 600      # Hueco inicial entre la pared izquierda y derecha
        self.MIN_GAP_SIZE = 120          # El hueco mínimo al que puede llegar el juego

        # --- Sistema de "Respawn" (Reaparición) ---
        self.AUTO_RESPAWN = True  # True: te recoloca en el centro. False: Game Over

        # --- Sistema de Dificultad Progresiva ---
        self.INCREASE_SPEED = True
        self.SPEED_INCREMENT_AMOUNT = 0.01   # Cuánto aumenta la velocidad de scroll
        self.SPEED_INCREMENT_INTERVAL = 300 # Cada cuántos ticks se aumenta (300 ticks = 5 seg a 60fps)

        self.DECREASE_GAP = True
        self.GAP_DECREASE_AMOUNT = 15       # Cuánto se reduce el hueco (espacio para pasar)
        self.GAP_DECREASE_INTERVAL = 300    # Cada cuántos ticks se reduce

        self.right = False
        self.left = False

        # =====================================================================

        # Configuración de ventana
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Endless Wave Runner - BCI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 32, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 20)

        self.reset_game()

    def reset_game(self):
        """
        Inicializa o reinicia todas las variables dinámicas del juego.
        Se llama al empezar o al pulsar 'ESC' tras un Game Over.
        """
        self.is_game_over = False
        self.score = 0.0
        self.ticks_survived = 0

        self.current_scroll_speed = self.INITIAL_SCROLL_SPEED
        self.current_gap_size = self.INITIAL_GAP_SIZE

        # Variables de la nave
        self.ship_x = self.SCREEN_WIDTH // 2
        self.ship_direction = 0  # -1 (Izquierda), 0 (Recto), 1 (Derecha)

        # Variables del fondo (para el efecto de scroll visual)
        self.bg_offset_y = 0.0

        # Variables de los obstáculos
        self.tunnel_points = []  # Almacena diccionarios con {'y', 'left_x', 'right_x'}
        self.TUNNEL_RESOLUTION_Y = 30  # Generamos un nuevo segmento de túnel cada 30 píxeles
        self.current_tunnel_center = self.SCREEN_WIDTH // 2

        # Llenamos la pantalla de túnel inicial (recto al principio para dar tiempo)
        for y in range(self.SCREEN_HEIGHT, -self.TUNNEL_RESOLUTION_Y, -self.TUNNEL_RESOLUTION_Y):
            self.tunnel_points.append({
                'y': float(y),
                'left_x': self.current_tunnel_center - (self.current_gap_size / 2),
                'right_x': self.current_tunnel_center + (self.current_gap_size / 2)
            })

    def update_difficulty(self):
        """
        Aumenta la velocidad y disminuye el tamaño del hueco basándose
        en el tiempo (ticks) transcurrido, según las constantes definidas.
        """
        # Aumentar velocidad
        if self.INCREASE_SPEED and self.ticks_survived % self.SPEED_INCREMENT_INTERVAL == 0:
            self.current_scroll_speed += self.SPEED_INCREMENT_AMOUNT

        # Reducir hueco
        if self.DECREASE_GAP and self.ticks_survived % self.GAP_DECREASE_INTERVAL == 0:
            if self.current_gap_size > self.MIN_GAP_SIZE:
                self.current_gap_size -= self.GAP_DECREASE_AMOUNT
                if self.current_gap_size < self.MIN_GAP_SIZE:
                    self.current_gap_size = self.MIN_GAP_SIZE

    def generate_tunnel_obstacles(self):
        """
        ALGORITMO DE GENERACIÓN SEGURO:
        Controla la generación del túnel de obstáculos por encima de la pantalla.
        Para asegurar que siempre sea jugable, calcula el desplazamiento máximo 'dx_max'
        que el jugador puede hacer horizontalmente mientras el mapa baja verticalmente.
        Mueve el centro del túnel de forma aleatoria sin superar ese límite, garantizando
        que el jugador siempre tenga tiempo de llegar de un punto al siguiente.
        """
        # Eliminar puntos que ya salieron por debajo de la pantalla
        self.tunnel_points = [p for p in self.tunnel_points if p['y'] <= self.SCREEN_HEIGHT + 50]

        # Mientras el punto más alto esté dentro de la pantalla, generar nuevos puntos más arriba
        while self.tunnel_points[-1]['y'] > 0:
            highest_point = self.tunnel_points[-1]
            new_y = highest_point['y'] - self.TUNNEL_RESOLUTION_Y

            # Cálculo de jugabilidad garantizada
            # ratio = velocidad_horizontal / velocidad_vertical
            # Esto nos dice cuántos píxeles en X podemos movernos por cada píxel en Y
            maneuver_ratio = self.SHIP_X_SPEED / self.current_scroll_speed

            # El desplazamiento máximo permitido en X para la resolución de Y actual
            max_horizontal_shift = self.TUNNEL_RESOLUTION_Y * maneuver_ratio

            # Aplicamos un margen de seguridad (ej. 80%) para no requerir reflejos robóticos
            safe_shift = max_horizontal_shift * 0.8

            # Mover el centro aleatoriamente, pero dentro de los límites seguros
            shift = random.uniform(-safe_shift, safe_shift)
            self.current_tunnel_center += shift

            # Evitar que el centro se salga de la pantalla
            half_gap = self.current_gap_size / 2
            if self.current_tunnel_center - half_gap < 0:
                self.current_tunnel_center = half_gap
            elif self.current_tunnel_center + half_gap > self.SCREEN_WIDTH:
                self.current_tunnel_center = self.SCREEN_WIDTH - half_gap

            self.tunnel_points.append({
                'y': new_y,
                'left_x': self.current_tunnel_center - half_gap,
                'right_x': self.current_tunnel_center + half_gap
            })

    def check_collisions(self):
        """
        Comprueba si la nave toca los bordes (si no hay obstáculos)
        o si colisiona con las paredes del túnel (si los obstáculos están activados).
        Si hay colisión, evalúa si debe hacer respawn o mostrar el Game Over.
        """
        collision_detected = False

        # 1. Caso sin obstáculos: Chocar con los bordes izquierdo/derecho
        if not self.ENABLE_OBSTACLES:
            if self.ship_x - self.SHIP_SIZE < 0 or self.ship_x + self.SHIP_SIZE > self.SCREEN_WIDTH:
                collision_detected = True

        # 2. Caso con obstáculos: Chocar con los pinchos generados
        else:
            # Buscar el segmento del túnel donde se encuentra la nave actualmente
            # Como la nave está en SHIP_FIXED_Y, buscamos los dos puntos Y que la rodean.
            left_bound, right_bound = 0, self.SCREEN_WIDTH
            for i in range(len(self.tunnel_points) - 1):
                p1 = self.tunnel_points[i]
                p2 = self.tunnel_points[i+1]

                # p1 tiene una Y mayor (más abajo) y p2 una Y menor (más arriba)
                if p2['y'] <= self.SHIP_FIXED_Y <= p1['y']:
                    # Hacemos una interpolación lineal simple para calcular el ancho exacto en la Y de la nave
                    t = (self.SHIP_FIXED_Y - p2['y']) / (p1['y'] - p2['y']) if p1['y'] != p2['y'] else 0
                    left_bound = p2['left_x'] + t * (p1['left_x'] - p2['left_x'])
                    right_bound = p2['right_x'] + t * (p1['right_x'] - p2['right_x'])
                    break

            # Comprobar si los bordes de la nave tocan los límites calculados
            if self.ship_x - self.SHIP_SIZE < left_bound or self.ship_x + self.SHIP_SIZE > right_bound:
                collision_detected = True

        # --- Gestión de la Colisión ---
        if collision_detected:
            if self.AUTO_RESPAWN:
                # Buscar la zona más segura (el centro matemático del túnel en esa coordenada)
                if self.ENABLE_OBSTACLES:
                    self.ship_x = (left_bound + right_bound) / 2
                else:
                    self.ship_x = self.SCREEN_WIDTH / 2
                # Apuntar recto para no chocar de inmediato
                #self.ship_direction = 0
                self.current_scroll_speed = self.INITIAL_SCROLL_SPEED
                self.current_gap_size = self.INITIAL_GAP_SIZE
            else:
                self.is_game_over = True

    def update(self):
        """
        Actualiza toda la lógica del juego: posiciones, puntos, colisiones y estado.
        """
        if self.is_game_over:
            return

        self.ticks_survived += 1

        # Actualizar puntos (por tiempo sobrevivido y velocidad)
        self.score += (self.current_scroll_speed / 10.0)

        self.update_difficulty()

        # Mover la nave horizontalmente
        self.ship_x += self.ship_direction * self.SHIP_X_SPEED

        # Mover el fondo y los obstáculos (Scroll)
        self.bg_offset_y = (self.bg_offset_y + self.current_scroll_speed) % 100

        for point in self.tunnel_points:
            point['y'] += self.current_scroll_speed

        if self.ENABLE_OBSTACLES:
            self.generate_tunnel_obstacles()

        self.check_collisions()

    def draw_ship(self):
        """
        Dibuja la nave (flecha/triángulo) en la pantalla.
        Su rotación depende de self.ship_direction.
        """
        # Puntos básicos para un triángulo apuntando hacia arriba
        base_points = [(0, -self.SHIP_SIZE), (-self.SHIP_SIZE, self.SHIP_SIZE), (self.SHIP_SIZE, self.SHIP_SIZE)]

        # Ángulo de rotación: 0 recto, 45 izq, -45 der (en grados visuales)
        angle = 0
        if self.ship_direction == -1: angle = 45
        elif self.ship_direction == 1: angle = -45

        rotated_points = []
        for x, y in base_points:
            vec = pygame.math.Vector2(x, y).rotate(-angle) # Rotar vector
            # Trasladar al punto físico de la nave
            rotated_points.append((self.ship_x + vec.x, self.SHIP_FIXED_Y + vec.y))

        pygame.draw.polygon(self.screen, (0, 255, 255), rotated_points)
        pygame.draw.polygon(self.screen, (255, 255, 255), rotated_points, 2) # Borde blanco

    def draw_background(self):
        """
        Dibuja un fondo con líneas de cuadrícula en movimiento vertical continuo
        para dar la sensación de avance, sin marear.
        """
        self.screen.fill((20, 20, 30))  # Color base oscuro

        # Líneas horizontales bajando
        for y in range(int(self.bg_offset_y) - 100, self.SCREEN_HEIGHT, 100):
            pygame.draw.line(self.screen, (40, 40, 50), (0, y), (self.SCREEN_WIDTH, y), 2)

        # Líneas verticales estáticas
        for x in range(0, self.SCREEN_WIDTH, 100):
            pygame.draw.line(self.screen, (40, 40, 50), (x, 0), (x, self.SCREEN_HEIGHT), 2)

    def draw_obstacles(self):
        """
        Dibuja los obstáculos laterales basándose en la lista de puntos del túnel.
        Crea polígonos irregulares rellenando la zona fuera del túnel hacia los bordes.
        """
        if not self.ENABLE_OBSTACLES or len(self.tunnel_points) < 2:
            return

        # Preparar polígono izquierdo: borde de pantalla izq -> puntos de la pared -> borde de pantalla izq
        left_poly = [(0, self.tunnel_points[0]['y'])]
        for p in self.tunnel_points:
            left_poly.append((p['left_x'], p['y']))
        left_poly.append((0, self.tunnel_points[-1]['y']))

        # Preparar polígono derecho: puntos pared der -> borde pantalla der
        right_poly = [(self.SCREEN_WIDTH, self.tunnel_points[0]['y'])]
        for p in self.tunnel_points:
            right_poly.append((p['right_x'], p['y']))
        right_poly.append((self.SCREEN_WIDTH, self.tunnel_points[-1]['y']))

        # Dibujar los polígonos (simulando pinchos/paredes oscuras y bordes rojos peligrosos)
        pygame.draw.polygon(self.screen, (50, 10, 10), left_poly)
        pygame.draw.polygon(self.screen, (50, 10, 10), right_poly)

        pygame.draw.lines(self.screen, (255, 50, 50), False, [(p['left_x'], p['y']) for p in self.tunnel_points], 3)
        pygame.draw.lines(self.screen, (255, 50, 50), False, [(p['right_x'], p['y']) for p in self.tunnel_points], 3)

    def draw_ui(self):
        """Dibuja la puntuación y las pantallas de estado (Game Over)."""
        # Score
        score_surface = self.font.render(f"Puntos: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_surface, (10, 10))

        # Info debug (opcional pero útil)
        info_str = f"Vel: {self.current_scroll_speed:.1f} | Hueco: {int(self.current_gap_size)}"
        info_surface = self.small_font.render(info_str, True, (200, 200, 200))
        self.screen.blit(info_surface, (10, 50))

        if self.is_game_over:
            # Oscurecer fondo ligeramente
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            overlay.set_alpha(150)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))

            # Texto
            go_text = self.font.render("GAME OVER", True, (255, 50, 50))
            go_rect = go_text.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 - 30))
            self.screen.blit(go_text, go_rect)

            restart_text = self.small_font.render("Pulsa 'ESC' para volver a jugar", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 + 20))
            self.screen.blit(restart_text, restart_rect)

    # ------------------------------------------------------------------
    # INTEGRACIÓN BCI
    # ------------------------------------------------------------------

    def string_to_action(self, prediction: str) -> None:
        """Traduce la predicción del pipe a una dirección de la nave."""
        if prediction == "left_hand":
            self.ship_direction = -1
        elif prediction == "right_hand":
            self.ship_direction = 1
        else:
            self.ship_direction = self.ship_direction #0 # NOTA: no se contempla la posibilidad de ir recto

    def controlAssist(self, prediction: str, info: dict) -> None:
        if AIMBOT_MODE:
            if info["name"] == "ExponentialSmoothing":
                self.heuristica_aimbot_2(prediction, info)
        else:
            self.string_to_action(prediction)

    def heuristica_aimbot_1(self, prediction, info): 
        left_bound, right_bound = 0, self.SCREEN_WIDTH
        for i in range(len(self.tunnel_points) - 1): # CALCULAMOS LOS BORDES IZQUIERDO Y DERECHO
            p1 = self.tunnel_points[i]
            p2 = self.tunnel_points[i+1]

            # p1 tiene una Y mayor (más abajo) y p2 una Y menor (más arriba)
            if p2['y'] <= self.SHIP_FIXED_Y <= p1['y']:
                # Hacemos una interpolación lineal simple para calcular el ancho exacto en la Y de la nave
                t = (self.SHIP_FIXED_Y - p2['y']) / (p1['y'] - p2['y']) if p1['y'] != p2['y'] else 0
                left_bound = p2['left_x'] + t * (p1['left_x'] - p2['left_x'])
                right_bound = p2['right_x'] + t * (p1['right_x'] - p2['right_x'])
                break

        tunnel_center = (left_bound + right_bound) / 2

        # formulas que vamos a aplicar para sacar el umbral normalizado
        # umbral_normalizado = umbral_min + (distancia / distancia_maxima) * (umbral_max - umbral_min)
        # y cuando queramos q 0 sea el max (para el lado derecho)
        # umbral_normalizado = umbral_max - (distancia / distancia_maxima) * (umbral_max - umbral_min)

        umbral_min = 1 - SEVERIDAD_AIMBOT
        umbral_max = 1

        # POR SI QUEDA MUY ANTINATURAL:
        # para lado izq:
        # tunnel_center -= self.current_gap_size/3
        # para lado der:
        # tunnel_center += self.current_gap_size/3
        # habría q adaptar el if tb:
        # if self.ship_x < (tunnel_center-self.current_gap_size/3)
        # elif self.ship_x > (tunnel_center+self.current_gap/3)
        # else //lo q digamos para el caso neutro//

        if self.ship_x < tunnel_center: # en principio hay q ir a la derecha
            left_dist = self.ship_x-left_bound
            max_dist = tunnel_center - left_bound
            umbral_normalizado = umbral_min + (left_dist / max_dist) * (umbral_max - umbral_min)
            if info["p_right_smooth"] > umbral_normalizado:
                self.ship_direction = 1
            else:
                self.string_to_action(prediction)
        else: # en principio hay q ir a la izquierda (ignoramos el caso en el q estamos justo en el centro)
            right_dist = right_bound-self.ship_x
            max_dist = right_bound - tunnel_center
            umbral_normalizado = umbral_max - (right_dist / max_dist) * (umbral_max - umbral_min)
            if info["p_left_smooth"] > umbral_normalizado:
                self.ship_direction = -1
            else:
                self.string_to_action(prediction)

    def heuristica_aimbot_2(self, prediction, info): 
        left_bound, right_bound = 0, self.SCREEN_WIDTH
        for i in range(len(self.tunnel_points) - 1): # CALCULAMOS LOS BORDES IZQUIERDO Y DERECHO
            p1 = self.tunnel_points[i]
            p2 = self.tunnel_points[i+1]

            # p1 tiene una Y mayor (más abajo) y p2 una Y menor (más arriba)
            if p2['y'] <= self.SHIP_FIXED_Y <= p1['y']:
                # Hacemos una interpolación lineal simple para calcular el ancho exacto en la Y de la nave
                t = (self.SHIP_FIXED_Y - p2['y']) / (p1['y'] - p2['y']) if p1['y'] != p2['y'] else 0
                left_bound = p2['left_x'] + t * (p1['left_x'] - p2['left_x'])
                right_bound = p2['right_x'] + t * (p1['right_x'] - p2['right_x'])
                break

        tunnel_center = (left_bound + right_bound) / 2

        # formulas que vamos a aplicar para sacar el umbral normalizado
        # umbral_normalizado = umbral_min + (distancia / distancia_maxima) * (umbral_max - umbral_min)
        # y cuando queramos q 0 sea el max (para el lado derecho)
        # umbral_normalizado = umbral_max - (distancia / distancia_maxima) * (umbral_max - umbral_min)

        umbral_min = 1 - SEVERIDAD_AIMBOT
        umbral_max = 1

        tunnel_center_left = tunnel_center - (self.current_gap_size/3)
        tunnel_center_right = tunnel_center + (self.current_gap_size/3)


        if self.ship_x < tunnel_center_left: # en principio hay q ir a la derecha
            self.right = True
            left_dist = self.ship_x-left_bound
            max_dist = tunnel_center_left - left_bound
            umbral_normalizado = umbral_min + (left_dist / max_dist) * (umbral_max - umbral_min)
            if info["p_right_smooth"] > umbral_normalizado:
                self.ship_direction = 1
            else:
                self.string_to_action(prediction)
        elif self.ship_x > tunnel_center_right: # en principio hay q ir a la izquierda (ignoramos el caso en el q estamos justo en el centro)
            self.left = True
            right_dist = right_bound-self.ship_x
            max_dist = right_bound - tunnel_center_right
            umbral_normalizado = umbral_max - (right_dist / max_dist) * (umbral_max - umbral_min)
            if info["p_left_smooth"] > umbral_normalizado:
                self.ship_direction = -1
            else:
                self.string_to_action(prediction)
        elif self.left and self.ship_x < tunnel_center:
            self.left = False
        elif self.right and self.ship_x > tunnel_center:
            self.right = False
        elif self.right:
            self.ship_direction = 1
        elif self.left:
            self.ship_direction = -1
        else:
            self.string_to_action(prediction)

    # ------------------------------------------------------------------
    # BUCLE PRINCIPAL
    # ------------------------------------------------------------------

    def start(self, filename: Optional[str] = None) -> None:
        """Bucle principal de ejecución del juego. Escucha el pipe BCI en cada frame."""
        running = True
        while running and not self.stop_event.is_set():
            # -- Eventos pygame --
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.is_game_over:
                            self.reset_game()
                        else:
                            running = False
                    elif not self.is_game_over:
                        if event.key == pygame.K_LEFT:
                            self.ship_direction = -1
                        elif event.key == pygame.K_RIGHT:
                            self.ship_direction = 1
                        elif event.key == pygame.K_UP:
                            self.ship_direction = 0

                # -- Eventos BCI inyectados desde el pipe --
                elif event.type == PREDICTED_LEFT and not self.is_game_over:
                    self.ship_direction = -1
                elif event.type == PREDICTED_RIGHT and not self.is_game_over:
                    self.ship_direction = 1
                elif event.type == PREDICTED_REST and not self.is_game_over:
                    self.ship_direction = 0

            # -- Lectura no bloqueante del pipe BCI: cada predicción nueva cambia la dirección --
            try:
                while self.pipe.poll(0):
                    try:
                        prediction, info = self.read_msg()
                        if not self.is_game_over:
                            self.controlAssist(prediction, info)
                    except Exception:
                        pass
            except OSError:
                running = False

            #self.controlAssist("right_hand", {"name":"ExponentialSmoothing", "p_right_smooth":100, "p_left_smooth":100})

            self.update()

            self.draw_background()
            self.draw_obstacles()
            self.draw_ship()
            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(self.TICK_RATE)

        pygame.quit()
        sys.exit()


# =====================================================================
# EJECUCIÓN STANDALONE (para pruebas sin pipeline BCI)
# =====================================================================
if __name__ == "__main__":
    from components.RT_pipe_components.MIGames.MI_game_process import MIGameProcess

    proc = MIGameProcess(EndlessWaveRunnerMIGame)
    proc.start()
    proc.process.join()

from __future__ import annotations

import math
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
SEVERIDAD_AIMBOT = 0


class ProjectArrowsMIGame(MIGame):
    """
    Juego tipo Project Rhombus adaptado a 2 direcciones con integración BCI.
    """

    def __init__(self, pipe: Connection, stop_event: Event) -> None:
        super().__init__(pipe, stop_event)
        pygame.init()

        # =====================================================================
        # CONSTANTES DE CONFIGURACIÓN DEL JUEGO (Completamente modificables)
        # =====================================================================

        # --- Pantalla y Rendimiento ---
        self.WIDTH = 800
        self.HEIGHT = 600
        self.FPS = 60
        self.CENTER_X = self.WIDTH // 2
        self.CENTER_Y = self.HEIGHT // 2

        # --- Constantes Solicitadas: Dificultad y Velocidad ---
        self.BASE_ENEMY_SPEED = 1.0          # Velocidad inicial base de las flechas (píxeles por frame aprox)
        self.INCREASE_SPEED_OVER_TIME = True # ¿Aumenta la velocidad con el tiempo?
        self.SPEED_INC_PER_0_01S = 0.0005     # Cuánto aumenta la velocidad por cada 0.01 segundos

        # --- Constantes Solicitadas: Mecánicas ---
        self.ALLOW_INVERTED_ARROWS = False    # Generar flechas invertidas que cambian de lado
        self.AUTO_RESPAWN = False            # True: Reinicia sin pantalla Game Over. False: Pantalla clásica.

        # --- Distancias y Comportamiento ---
        self.SWAP_DISTANCE = 250             # Distancia al centro donde la flecha invertida empieza a cambiar de lado
        self.ARC_HEIGHT = 150                # Altura máxima del salto visual de la flecha invertida
        self.HITBOX_RADIUS = 35              # Distancia al centro para considerar que la flecha ha impactado
        self.BASE_SPAWN_RATE = 6000          # Tiempo base de aparición (en milisegundos)

        # =====================================================================

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Project Arrows - BCI")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("Verdana", 48, bold=True)
        self.font_score = pygame.font.SysFont("Courier New", 36, bold=True)
        self.font_info = pygame.font.SysFont("Verdana", 18)

        self.state = "PLAYING"
        self.reset_game()

    def reset_game(self):
        """Reinicia todos los valores dinámicos para una nueva partida."""
        self.player_dir = 1  # 1 (Apunta Derecha), -1 (Apunta Izquierda)
        self.enemies = []

        self.current_speed = self.BASE_ENEMY_SPEED
        self.survived_time = 0.0
        self.time_accumulator = 0.0
        self.spawn_timer = 0.0

        self.state = "PLAYING"

    def spawn_enemy(self):
        """Genera una nueva flecha en los bordes de la pantalla."""
        side = random.choice([-1, 1])
        is_inverted = self.ALLOW_INVERTED_ARROWS and (random.random() < 0.25)
        start_x = -50 if side == -1 else self.WIDTH + 50

        self.enemies.append({
            'x': start_x,
            'y': self.CENTER_Y,
            'side': side,
            'is_inverted': is_inverted,
            'is_swapping': False,
            'has_swapped': False,
            'swap_t': 0.0,
            'color': (255, 50, 150) if is_inverted else (50, 255, 255)
        })

    def update_logic(self, dt):
        """
        Procesa la lógica del juego frame a frame.
        dt = delta time (milisegundos transcurridos desde el último frame).
        """
        dt_sec = dt / 1000.0
        self.survived_time += dt_sec
        self.time_accumulator += dt_sec

        if self.INCREASE_SPEED_OVER_TIME:
            while self.time_accumulator >= 0.01:
                self.current_speed += self.SPEED_INC_PER_0_01S
                self.time_accumulator -= 0.01

        self.spawn_timer += dt
        current_spawn_rate = self.BASE_SPAWN_RATE * (self.BASE_ENEMY_SPEED / self.current_speed)

        if self.spawn_timer >= current_spawn_rate:
            self.spawn_enemy()
            self.spawn_timer = 0

        move_multiplier = dt / 16.66

        for enemy in self.enemies[:]:
            if enemy['is_inverted'] and not enemy['has_swapped']:
                dist_to_center = abs(enemy['x'] - self.CENTER_X)
                if dist_to_center <= self.SWAP_DISTANCE and not enemy['is_swapping']:
                    enemy['is_swapping'] = True
                    enemy['x'] = self.CENTER_X + (enemy['side'] * self.SWAP_DISTANCE)

            if enemy['is_swapping']:
                salto_speed = (self.current_speed * 10) / (self.SWAP_DISTANCE * 1)
                enemy['swap_t'] += salto_speed * move_multiplier

                if enemy['swap_t'] >= 1.0:
                    enemy['is_swapping'] = False
                    enemy['has_swapped'] = True
                    enemy['side'] *= -1
                    enemy['x'] = self.CENTER_X + (enemy['side'] * self.SWAP_DISTANCE)
                    enemy['y'] = self.CENTER_Y
                else:
                    start_x = self.CENTER_X + (enemy['side'] * self.SWAP_DISTANCE)
                    end_x = self.CENTER_X - (enemy['side'] * self.SWAP_DISTANCE)
                    enemy['x'] = start_x + (end_x - start_x) * enemy['swap_t']
                    enemy['y'] = self.CENTER_Y - math.sin(enemy['swap_t'] * math.pi) * self.ARC_HEIGHT
            else:
                enemy['x'] -= enemy['side'] * self.current_speed * move_multiplier

            if not enemy['is_swapping'] and abs(enemy['x'] - self.CENTER_X) <= self.HITBOX_RADIUS:
                if self.player_dir == enemy['side']:
                    self.enemies.remove(enemy)
                else:
                    if self.AUTO_RESPAWN:
                        self.reset_game()
                    else:
                        self.state = "GAME_OVER"
                    break

    def draw_arrow(self, surface, x, y, direction, size, color, is_inverted=False):
        """Dibuja una flecha geométrica."""
        vis_dir = direction * (-1 if is_inverted else 1)

        p1 = (x + (size * vis_dir), y)
        p2 = (x - (size * vis_dir), y - size*0.8)
        p3 = (x - (size * 0.4 * vis_dir), y)
        p4 = (x - (size * vis_dir), y + size*0.8)

        pygame.draw.polygon(surface, color, [p1, p2, p3, p4])
        pygame.draw.polygon(surface, (255, 255, 255), [p1, p2, p3, p4], 2)

    def draw(self):
        """Dibuja todos los elementos en la pantalla."""
        self.screen.fill((15, 18, 25))

        pygame.draw.line(self.screen, (30, 35, 45), (0, self.CENTER_Y), (self.WIDTH, self.CENTER_Y), 2)
        pygame.draw.circle(self.screen, (30, 35, 45), (self.CENTER_X, self.CENTER_Y), self.HITBOX_RADIUS, 2)

        if self.ALLOW_INVERTED_ARROWS:
            c = (100, 30, 80)
            pygame.draw.line(self.screen, c, (self.CENTER_X - self.SWAP_DISTANCE, self.CENTER_Y - 30), (self.CENTER_X - self.SWAP_DISTANCE, self.CENTER_Y + 30), 2)
            pygame.draw.line(self.screen, c, (self.CENTER_X + self.SWAP_DISTANCE, self.CENTER_Y - 30), (self.CENTER_X + self.SWAP_DISTANCE, self.CENTER_Y + 30), 2)

        self.draw_arrow(self.screen, self.CENTER_X, self.CENTER_Y, self.player_dir, 30, (255, 255, 255))

        for e in self.enemies:
            base_dir = -e['side']
            vis_inv = e['is_inverted'] and not e['has_swapped']
            self.draw_arrow(self.screen, e['x'], e['y'], base_dir, 20, e['color'], vis_inv)

        time_str = f"{self.survived_time:05.2f}"
        score_surf = self.font_score.render(time_str, True, (220, 220, 220))
        self.screen.blit(score_surf, (self.WIDTH//2 - score_surf.get_width()//2, 40))

        if self.state in ["PAUSED", "GAME_OVER"]:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
            overlay.set_alpha(200)
            overlay.fill((10, 10, 15))
            self.screen.blit(overlay, (0, 0))

            if self.state == "PAUSED":
                t1 = self.font_title.render("PAUSADO", True, (255, 255, 255))
                t2 = self.font_score.render(f"Tiempo: {time_str}", True, (150, 200, 255))
                t3 = self.font_info.render("Pulsa ESC para continuar", True, (150, 150, 150))
            else:
                t1 = self.font_title.render("GAME OVER", True, (255, 80, 80))
                t2 = self.font_score.render(f"Supervivencia: {time_str}s", True, (255, 255, 255))
                t3 = self.font_info.render("Pulsa ESC para reiniciar", True, (150, 150, 150))

            self.screen.blit(t1, (self.WIDTH//2 - t1.get_width()//2, self.CENTER_Y - 90))
            self.screen.blit(t2, (self.WIDTH//2 - t2.get_width()//2, self.CENTER_Y - 10))
            self.screen.blit(t3, (self.WIDTH//2 - t3.get_width()//2, self.CENTER_Y + 60))

    # ------------------------------------------------------------------
    # INTEGRACIÓN BCI
    # ------------------------------------------------------------------

    def string_to_action(self, prediction: str) -> None:
        """Traduce la predicción del pipe a una dirección del jugador."""
        if prediction == "left_hand":
            self.player_dir = -1
        elif prediction == "right_hand":
            self.player_dir = 1

    def controlAssist(self, prediction: str, info: dict) -> None:
        if AIMBOT_MODE:
            if info["name"] == "ExponentialSmoothing":
                self.heuristica_aimbot_1(prediction, info)
        else:
            self.string_to_action(prediction)

    def heuristica_aimbot_1(self, prediction, info):
        enemigo_mas_cercano = None
        distancia_minima = float('inf')

        for enemy in self.enemies: # Sacamos el enemigo más cercano y su distancia
            # La distancia absoluta desde la posición X del enemigo hasta nuestro centro
            distancia = abs(enemy['x'] - self.CENTER_X)
            
            if distancia < distancia_minima:
                distancia_minima = distancia
                enemigo_mas_cercano = enemy

        if enemigo_mas_cercano:
            lado_origen = enemigo_mas_cercano['side']
            
            es_engano = enemigo_mas_cercano['is_inverted'] and not enemigo_mas_cercano['has_swapped'] # para comprobar si la flecha va a cambiar de lado

            if es_engano:
                lado_real_impacto = lado_origen * -1
            else:
                lado_real_impacto = lado_origen

            umbral = 1 - SEVERIDAD_AIMBOT

            if lado_real_impacto == -1: # deberíamos apuntar a la izquierda
                if info["p_right_smooth"] >= umbral:
                    self.player_dir = -1
                else:
                    self.string_to_action(prediction)
            elif lado_real_impacto == 1: # deberíamos apuntar a la derecha
                if info["p_left_smooth"] >= umbral:
                    self.player_dir = 1
                else:
                    self.string_to_action(prediction)

    # ------------------------------------------------------------------
    # BUCLE PRINCIPAL
    # ------------------------------------------------------------------

    def start(self, _filename: Optional[str] = None) -> None:
        """Bucle principal de ejecución del juego. Escucha el pipe BCI en cada frame."""
        running = True
        while running and not self.stop_event.is_set():
            dt = self.clock.tick(self.FPS)

            # -- Eventos pygame --
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if self.state == "PLAYING":
                        if event.key == pygame.K_LEFT:
                            self.player_dir = -1
                        elif event.key == pygame.K_RIGHT:
                            self.player_dir = 1
                        elif event.key == pygame.K_ESCAPE:
                            self.state = "PAUSED"

                    elif self.state == "PAUSED":
                        if event.key == pygame.K_ESCAPE:
                            self.state = "PLAYING"

                    elif self.state == "GAME_OVER":
                        if event.key == pygame.K_ESCAPE:
                            self.reset_game()

                # -- Eventos BCI inyectados desde el pipe --
                elif event.type == PREDICTED_LEFT and self.state == "PLAYING":
                    self.player_dir = -1
                elif event.type == PREDICTED_RIGHT and self.state == "PLAYING":
                    self.player_dir = 1

            # -- Lectura no bloqueante del pipe BCI --
            try:
                while self.pipe.poll(0):
                    try:
                        prediction, info = self.read_msg()
                        if self.state == "PLAYING":
                            self.controlAssist(prediction, info)
                    except Exception:
                        pass
            except OSError:
                running = False
            
            #self.controlAssist("right_hand", {"name":"ExponentialSmoothing", "p_right_smooth":100, "p_left_smooth":100})

            if self.state == "PLAYING":
                self.update_logic(dt)

            self.draw()
            pygame.display.flip()

        pygame.quit()
        sys.exit()


# =====================================================================
# EJECUCIÓN STANDALONE (para pruebas sin pipeline BCI)
# =====================================================================
if __name__ == "__main__":
    from components.RT_pipe_components.MIGames.MI_game_process import MIGameProcess

    proc = MIGameProcess(ProjectArrowsMIGame)
    proc.start()
    proc.process.join()

import pygame
import sys
import random
import math

class ProjectArrows:
    """
    Clase principal que contiene el juego tipo Project Rhombus adaptado a 2 direcciones.
    """
    def __init__(self):
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
        self.BASE_ENEMY_SPEED = 5.0          # Velocidad inicial base de las flechas (píxeles por frame aprox)
        self.INCREASE_SPEED_OVER_TIME = True # ¿Aumenta la velocidad con el tiempo?
        self.SPEED_INC_PER_0_01S = 0.002     # Cuánto aumenta la velocidad por cada 0.01 segundos
        
        # --- Constantes Solicitadas: Mecánicas ---
        self.ALLOW_INVERTED_ARROWS = True    # Generar flechas invertidas que cambian de lado
        self.AUTO_RESPAWN = False            # True: Reinicia sin pantalla Game Over. False: Pantalla clásica.
        
        # --- Distancias y Comportamiento ---
        self.SWAP_DISTANCE = 250             # Distancia al centro donde la flecha invertida empieza a cambiar de lado
        self.ARC_HEIGHT = 150                # Altura máxima del salto visual de la flecha invertida
        self.HITBOX_RADIUS = 35              # Distancia al centro para considerar que la flecha ha impactado
        self.BASE_SPAWN_RATE = 1000          # Tiempo base de aparición (en milisegundos)
        
        # =====================================================================
        
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Project Arrows - Reflex Runner")
        self.clock = pygame.time.Clock()
        
        # Fuentes de texto
        self.font_title = pygame.font.SysFont("Verdana", 48, bold=True)
        self.font_score = pygame.font.SysFont("Courier New", 36, bold=True)
        self.font_info = pygame.font.SysFont("Verdana", 18)
        
        # Estado principal
        self.state = "PLAYING" # Puede ser "PLAYING", "PAUSED", "GAME_OVER"
        self.reset_game()

    def reset_game(self):
        """
        Reinicia todos los valores dinámicos para una nueva partida.
        """
        self.player_dir = 1  # 1 (Apunta Derecha), -1 (Apunta Izquierda)
        self.enemies = []
        
        self.current_speed = self.BASE_ENEMY_SPEED
        self.survived_time = 0.0    # Tiempo en segundos
        self.time_accumulator = 0.0 # Acumulador para la lógica de los 0.01s
        self.spawn_timer = 0.0      # Temporizador para la generación de enemigos
        
        self.state = "PLAYING"

    def spawn_enemy(self):
        """
        Genera una nueva flecha en los bordes de la pantalla.
        Controla si es invertida en base a las constantes y el azar.
        """
        side = random.choice([-1, 1]) # -1: Viene de la Izquierda. 1: Viene de la Derecha.
        
        # 25% de probabilidad de ser invertida si están activadas
        is_inverted = self.ALLOW_INVERTED_ARROWS and (random.random() < 0.25)
        
        # Empezamos fuera de la pantalla
        start_x = -50 if side == -1 else self.WIDTH + 50
        
        self.enemies.append({
            'x': start_x,
            'y': self.CENTER_Y,
            'side': side,
            'is_inverted': is_inverted,
            'is_swapping': False,    # True cuando está en el aire cambiando de lado
            'has_swapped': False,    # True cuando ya ha aterrizado en el lado opuesto
            'swap_t': 0.0,           # Progreso del arco de salto (de 0.0 a 1.0)
            'color': (255, 50, 150) if is_inverted else (50, 255, 255) # Magenta para Invertida, Cian para Normal
        })

    def update_logic(self, dt):
        """
        Procesa la lógica del juego frame a frame.
        dt = delta time (milisegundos transcurridos desde el último frame).
        """
        # 1. Actualizar tiempos
        dt_sec = dt / 1000.0
        self.survived_time += dt_sec
        self.time_accumulator += dt_sec
        
        # 2. Aumentar Velocidad cada 0.01s
        if self.INCREASE_SPEED_OVER_TIME:
            while self.time_accumulator >= 0.01:
                self.current_speed += self.SPEED_INC_PER_0_01S
                self.time_accumulator -= 0.01
                
        # 3. Generación de Enemigos
        self.spawn_timer += dt
        # Acortamos el tiempo de spawn según sube la velocidad para no dejar huecos enormes
        current_spawn_rate = self.BASE_SPAWN_RATE * (self.BASE_ENEMY_SPEED / self.current_speed)
        
        if self.spawn_timer >= current_spawn_rate:
            self.spawn_enemy()
            self.spawn_timer = 0
            
        # 4. Actualizar estado y posición de enemigos
        # Usamos dt/16.66 para normalizar el movimiento como si estuviera a 60 FPS fijos
        move_multiplier = dt / 16.66 
        
        for enemy in self.enemies[:]:
            # Lógica de salto para flechas invertidas
            if enemy['is_inverted'] and not enemy['has_swapped']:
                dist_to_center = abs(enemy['x'] - self.CENTER_X)
                
                # Inicia el salto
                if dist_to_center <= self.SWAP_DISTANCE and not enemy['is_swapping']:
                    enemy['is_swapping'] = True
                    # Encajamos perfectamente la X al inicio del arco para evitar tirones
                    enemy['x'] = self.CENTER_X + (enemy['side'] * self.SWAP_DISTANCE) 

            if enemy['is_swapping']:
                """
                ALGORITMO DE SALTO EN ARCO:
                Utilizamos una interpolación (t de 0 a 1) para mover la 'x' del punto de origen al destino.
                A la vez, la 'y' dibuja una función seno (parábola) para dar un arco visual fluido.
                """
                # La velocidad del salto depende de la velocidad del juego
                salto_speed = (self.current_speed * 1.5) / (self.SWAP_DISTANCE * 2)
                enemy['swap_t'] += salto_speed * move_multiplier
                
                if enemy['swap_t'] >= 1.0:
                    # Finaliza el salto
                    enemy['is_swapping'] = False
                    enemy['has_swapped'] = True
                    enemy['side'] *= -1 # Ahora viene del lado opuesto
                    enemy['x'] = self.CENTER_X + (enemy['side'] * self.SWAP_DISTANCE)
                    enemy['y'] = self.CENTER_Y
                else:
                    # Interpolar X
                    start_x = self.CENTER_X + (enemy['side'] * self.SWAP_DISTANCE)
                    end_x = self.CENTER_X - (enemy['side'] * self.SWAP_DISTANCE)
                    enemy['x'] = start_x + (end_x - start_x) * enemy['swap_t']
                    
                    # Interpolar Y (Arco hacia arriba)
                    enemy['y'] = self.CENTER_Y - math.sin(enemy['swap_t'] * math.pi) * self.ARC_HEIGHT
            else:
                # Movimiento normal hacia el centro
                enemy['x'] -= enemy['side'] * self.current_speed * move_multiplier

            # 5. Colisiones en el centro (Solo si no está saltando por encima nuestro)
            if not enemy['is_swapping'] and abs(enemy['x'] - self.CENTER_X) <= self.HITBOX_RADIUS:
                # 'side' indica de dónde viene. Si viene de la izquierda es -1.
                # Para bloquearla, el jugador debe mirar hacia la izquierda (player_dir == -1).
                if self.player_dir == enemy['side']:
                    # Bloqueo exitoso
                    self.enemies.remove(enemy)
                else:
                    # Muerte
                    if self.AUTO_RESPAWN:
                        self.reset_game()
                    else:
                        self.state = "GAME_OVER"
                    break

    def handle_events(self):
        """Manejo de entradas de teclado."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN:
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

    def draw_arrow(self, surface, x, y, direction, size, color, is_inverted=False):
        """
        Dibuja una flecha geométrica.
        - direction: 1 (derecha), -1 (izquierda).
        - is_inverted: Gira la flecha visualmente para que apunte al revés de su dirección natural.
        """
        vis_dir = direction * (-1 if is_inverted else 1)
        
        # Vértices base de la flecha
        p1 = (x + (size * vis_dir), y)             # Punta
        p2 = (x - (size * vis_dir), y - size*0.8)  # Esquina trasera arriba
        p3 = (x - (size * 0.4 * vis_dir), y)       # Hendidura trasera
        p4 = (x - (size * vis_dir), y + size*0.8)  # Esquina trasera abajo
        
        pygame.draw.polygon(surface, color, [p1, p2, p3, p4])
        pygame.draw.polygon(surface, (255, 255, 255), [p1, p2, p3, p4], 2)

    def draw(self):
        """Dibuja todos los elementos en la pantalla."""
        self.screen.fill((15, 18, 25)) # Fondo azul oscuro muy sutil
        
        # Decoración sutil del horizonte
        pygame.draw.line(self.screen, (30, 35, 45), (0, self.CENTER_Y), (self.WIDTH, self.CENTER_Y), 2)
        pygame.draw.circle(self.screen, (30, 35, 45), (self.CENTER_X, self.CENTER_Y), self.HITBOX_RADIUS, 2)
        
        # Zonas de salto (líneas guía visuales)
        if self.ALLOW_INVERTED_ARROWS:
            c = (100, 30, 80)
            pygame.draw.line(self.screen, c, (self.CENTER_X - self.SWAP_DISTANCE, self.CENTER_Y - 30), (self.CENTER_X - self.SWAP_DISTANCE, self.CENTER_Y + 30), 2)
            pygame.draw.line(self.screen, c, (self.CENTER_X + self.SWAP_DISTANCE, self.CENTER_Y - 30), (self.CENTER_X + self.SWAP_DISTANCE, self.CENTER_Y + 30), 2)

        # Dibujar Jugador
        self.draw_arrow(self.screen, self.CENTER_X, self.CENTER_Y, self.player_dir, 30, (255, 255, 255))
        
        # Dibujar Enemigos
        for e in self.enemies:
            # La dirección base es contraria al lado de donde vienen, para apuntar al centro
            base_dir = -e['side']
            # Está "visualmente invertida" si tiene la etiqueta y NO ha saltado todavía
            vis_inv = e['is_inverted'] and not e['has_swapped']
            
            self.draw_arrow(self.screen, e['x'], e['y'], base_dir, 20, e['color'], vis_inv)

        # Texto Puntuación (Formato 00.00)
        time_str = f"{self.survived_time:05.2f}"
        score_surf = self.font_score.render(time_str, True, (220, 220, 220))
        self.screen.blit(score_surf, (self.WIDTH//2 - score_surf.get_width()//2, 40))

        # Overlays para menús
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

    def run(self):
        """Bucle principal de la aplicación."""
        while True:
            dt = self.clock.tick(self.FPS)
            self.handle_events()
            
            if self.state == "PLAYING":
                self.update_logic(dt)
                
            self.draw()
            pygame.display.flip()

# =====================================================================
# EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    juego = ProjectArrows()
    juego.run()
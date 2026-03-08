import pygame
import random
import sys
import time

# --- COLORES ---
NEGRO = (0, 0, 0)
BLANCO = (255, 255, 255)
VERDE = (50, 205, 50)     # Jugador
ROJO = (220, 20, 60)      # Obstáculos
GRIS = (50, 50, 50)       # Líneas de cuadrícula
AZUL = (100, 149, 237)    # Meta / Nivel completado

# --- CONFIGURACIÓN POR DEFECTO ---
ANCHO_VENTANA = 600
ALTO_VENTANA = 800
FILAS = 10 

class Configuracion:
    def __init__(self):
        print("--- CONFIGURACIÓN DEL JUEGO BCI ---")
        self.columnas = self.pedir_numero("Número de carriles (1-5): ", 1, 5)
        self.obstaculos = self.pedir_si_no("¿Incluir obstáculos? (s/n): ")
        self.avance_auto = self.pedir_si_no("¿Avance automático? (s/n): ")
        self.respawn = self.pedir_si_no("¿Respawn automático al chocar? (s/n): ")
        
        if self.avance_auto:
            self.velocidad = 1000 # milisegundos por paso (INICIAL)
        else:
            self.velocidad = 0

    def pedir_numero(self, texto, min_val, max_val):
        while True:
            try:
                val = int(input(texto))
                if min_val <= val <= max_val:
                    return val
            except ValueError:
                pass
            print(f"Por favor, introduce un número entre {min_val} y {max_val}.")

    def pedir_si_no(self, texto):
        while True:
            val = input(texto).lower()
            if val in ['s', 'si', 'y', 'yes']: return True
            if val in ['n', 'no']: return False

class Juego:
    def __init__(self, config):
        pygame.init()
        self.cfg = config
        self.pantalla = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
        pygame.display.set_caption("BCI Arrow Runner (Completable)")
        self.reloj = pygame.time.Clock()
        self.fuente = pygame.font.SysFont("Arial", 24)
        self.fuente_grande = pygame.font.SysFont("Arial", 60)

        self.ancho_celda = ANCHO_VENTANA // self.cfg.columnas
        self.alto_celda = ALTO_VENTANA // FILAS

        self.jugador_x = self.cfg.columnas // 2 
        self.jugador_y = FILAS - 1              
        self.puntuacion = 0
        self.nivel = 1
        self.mapa_obstaculos = [] 
        self.camino_seguro = set() 
        self.game_over = False
        
        self.ultimo_tiempo_mov = pygame.time.get_ticks()

        self.generar_obstaculos()

    def generar_obstaculos(self):
        self.mapa_obstaculos = []
        self.camino_seguro = set()
        
        if not self.cfg.obstaculos:
            return

        curr_x = self.cfg.columnas // 2
        
        # Algoritmo de camino seguro
        for y in range(FILAS - 1, -1, -1):
            self.camino_seguro.add((curr_x, y))
            move = random.choice([-1, 0, 1]) 
            next_x = curr_x + move
            next_x = max(0, min(next_x, self.cfg.columnas - 1))
            
            if next_x != curr_x:
                self.camino_seguro.add((next_x, y))
            curr_x = next_x

        # Relleno de obstáculos
        for y in range(FILAS - 2): 
            for x in range(self.cfg.columnas):
                if (x, y) not in self.camino_seguro:
                    if random.random() < 0.5: 
                        self.mapa_obstaculos.append((x, y))

    def mover_jugador(self, dx, dy):
        if self.game_over: return

        nueva_x = self.jugador_x + dx
        nueva_y = self.jugador_y + dy

        if nueva_x < 0 or nueva_x >= self.cfg.columnas:
            return 

        if nueva_y < 0:
            self.nivel_completado()
            return

        chocado = False
        if (nueva_x, nueva_y) in self.mapa_obstaculos:
            chocado = True
        
        if chocado:
            if self.cfg.respawn:
                print("¡Muro! (Respawn activo)")
                self.pantalla.fill(ROJO)
                pygame.display.flip()
                time.sleep(0.05)
            else:
                self.game_over = True
        else:
            self.jugador_x = nueva_x
            self.jugador_y = nueva_y
            if dy < 0:
                self.puntuacion += 1

    def nivel_completado(self):
        self.nivel += 1
        self.puntuacion += 5 
        
        # --- AUMENTO DE VELOCIDAD ---
        if self.cfg.avance_auto:
            # Restamos 50ms al intervalo (más rápido) pero nunca bajamos de 200ms
            self.cfg.velocidad = max(200, self.cfg.velocidad - 50)
            print(f"¡Velocidad aumentada! Nuevo intervalo: {self.cfg.velocidad}ms")

        self.jugador_y = FILAS - 1 
        self.jugador_x = self.cfg.columnas // 2 
        self.generar_obstaculos()
        
        self.pantalla.fill(AZUL)
        texto = self.fuente_grande.render(f"NIVEL {self.nivel}", True, BLANCO)
        rect = texto.get_rect(center=(ANCHO_VENTANA//2, ALTO_VENTANA//2))
        self.pantalla.blit(texto, rect)
        
        # Mostrar velocidad actual si es automático
        if self.cfg.avance_auto:
            vel_txt = self.fuente.render(f"Velocidad: {self.cfg.velocidad}ms", True, BLANCO)
            vel_rect = vel_txt.get_rect(center=(ANCHO_VENTANA//2, ALTO_VENTANA//2 + 50))
            self.pantalla.blit(vel_txt, vel_rect)

        pygame.display.flip()
        time.sleep(1) 

    def dibujar_flecha(self):
        cx = self.jugador_x * self.ancho_celda + self.ancho_celda // 2
        cy = self.jugador_y * self.alto_celda + self.alto_celda // 2
        size = min(self.ancho_celda, self.alto_celda) // 3
        puntos = [
            (cx, cy - size),           
            (cx - size, cy + size),    
            (cx + size, cy + size)     
        ]
        pygame.draw.polygon(self.pantalla, VERDE, puntos)

    def dibujar(self):
        # 1. Limpiar pantalla siempre
        self.pantalla.fill(NEGRO)

        if self.game_over:
            # --- PANTALLA GAME OVER (Fondo Negro Puro) ---
            texto = self.fuente_grande.render("GAME OVER", True, ROJO)
            rect = texto.get_rect(center=(ANCHO_VENTANA//2, ALTO_VENTANA//2))
            self.pantalla.blit(texto, rect)
            
            puntos_txt = self.fuente.render(f"Puntuación Final: {self.puntuacion}", True, BLANCO)
            puntos_rect = puntos_txt.get_rect(center=(ANCHO_VENTANA//2, ALTO_VENTANA//2 + 50))
            self.pantalla.blit(puntos_txt, puntos_rect)

            subtexto = self.fuente.render("Presiona ESC para salir", True, GRIS)
            subrect = subtexto.get_rect(center=(ANCHO_VENTANA//2, ALTO_VENTANA//2 + 100))
            self.pantalla.blit(subtexto, subrect)
        
        else:
            # --- JUEGO NORMAL ---
            # Cuadrícula
            for x in range(0, ANCHO_VENTANA, self.ancho_celda):
                pygame.draw.line(self.pantalla, GRIS, (x, 0), (x, ALTO_VENTANA))
            for y in range(0, ALTO_VENTANA, self.alto_celda):
                pygame.draw.line(self.pantalla, GRIS, (0, y), (ANCHO_VENTANA, y))

            # Obstáculos
            for obs_x, obs_y in self.mapa_obstaculos:
                rect = (
                    obs_x * self.ancho_celda + 5, 
                    obs_y * self.alto_celda + 5, 
                    self.ancho_celda - 10, 
                    self.alto_celda - 10
                )
                pygame.draw.rect(self.pantalla, ROJO, rect)

            self.dibujar_flecha()

            # UI
            info_score = self.fuente.render(f"Puntos: {self.puntuacion}", True, BLANCO)
            info_level = self.fuente.render(f"Nivel: {self.nivel}", True, BLANCO)
            self.pantalla.blit(info_score, (10, ALTO_VENTANA - 40))
            self.pantalla.blit(info_level, (ANCHO_VENTANA - 100, ALTO_VENTANA - 40))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    if not self.game_over:
                        if self.cfg.columnas > 1:
                            if event.key == pygame.K_a:
                                self.mover_jugador(-1, 0)
                            elif event.key == pygame.K_d:
                                self.mover_jugador(1, 0)
                        
                        if not self.cfg.avance_auto:
                            if event.key == pygame.K_w:
                                self.mover_jugador(0, -1) 

            if self.cfg.avance_auto and not self.game_over:
                ahora = pygame.time.get_ticks()
                if ahora - self.ultimo_tiempo_mov > self.cfg.velocidad:
                    self.mover_jugador(0, -1)
                    self.ultimo_tiempo_mov = ahora

            self.dibujar()
            self.reloj.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    config = Configuracion()
    juego = Juego(config)
    juego.run()
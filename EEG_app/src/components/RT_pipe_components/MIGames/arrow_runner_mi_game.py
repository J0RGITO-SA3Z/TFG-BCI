from __future__ import annotations
import pygame
import random
import sys
import time

import random
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from typing import Optional

import pygame

from components.RT_pipe_components.MIGames.MIGame import MIGame

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
COLUMNAS = 3 

PREDICTED_LEFT = pygame.USEREVENT
PREDICTED_RIGHT = pygame.USEREVENT

AIMBOT_MODE = True
SEVERIDAD_AIMBOT = 1 # 0.0 (sin ayuda) a 1.0 (ayuda total)

VELOCIDAD_INICIAL = 2000

class Configuracion:
    def __init__(self):
        print("--- CONFIGURACIÓN DEL JUEGO BCI ---")
        self.columnas = COLUMNAS #self.pedir_numero("Número de carriles (1-5): ", 1, 5)
        self.obstaculos = 's' #self.pedir_si_no("¿Incluir obstáculos? (s/n): ")
        self.avance_auto = 's' #self.pedir_si_no("¿Avance automático? (s/n): ")
        self.respawn = 's' #self.pedir_si_no("¿Respawn automático al chocar? (s/n): ")
        
        if self.avance_auto:
            self.velocidad = VELOCIDAD_INICIAL #self.pedir_numero("¿Cada cuántos ms da un paso? (s/n): ", 0, 10000) # 1000 milisegundos por paso (INICIAL)
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

class ArrowRunnerMIGame(MIGame):
    def __init__(self, pipe: Connection, stop_event: Event) -> None:
        super().__init__(pipe, stop_event)
        pygame.init()
        self.cfg = Configuracion()
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

        # ------------ AIMBOT ------------------
        self.posiciones_seguras = dict()
        self.matriz_obstaculos = [[False] * FILAS for _ in range(COLUMNAS)]

        self.generar_obstaculos()



    def generar_obstaculos(self):
        self.mapa_obstaculos = []
        self.camino_seguro = set()
        self.posiciones_seguras = dict()
        
        if not self.cfg.obstaculos:
            return

        curr_x = self.cfg.columnas // 2
        
        # Algoritmo de camino seguro
        for y in range(FILAS - 1, -1, -1):
            self.camino_seguro.add((curr_x, y))
            self.posiciones_seguras[y] = curr_x
            move = random.choice([-1, 0, 1]) 
            next_x = curr_x + move
            next_x = max(0, min(next_x, self.cfg.columnas - 1))
            
            if next_x != curr_x:
                self.camino_seguro.add((next_x, y))
                self.posiciones_seguras[y] = next_x
            curr_x = next_x


        self.matriz_obstaculos = [[False] * FILAS for _ in range(COLUMNAS)]
        # Relleno de obstáculos
        for y in range(FILAS - 2): 
            for x in range(self.cfg.columnas):
                if (x, y) not in self.camino_seguro:
                    if random.random() < 1: 
                        self.mapa_obstaculos.append((x, y))
                        self.matriz_obstaculos[x][y] = True

        #print(f"[DEBUG] Camino seguro: {self.camino_seguro}")
        #print(f"[DEBUG] Posiciones seguras: {self.posiciones_seguras}")
        #print(f"[DEBUG] Matriz obstaculos: {self.matriz_obstaculos}")

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
            self.cfg.velocidad = max(300, self.cfg.velocidad - 50)
            print(f"¡Velocidad aumentada! Nuevo intervalo: {self.cfg.velocidad}ms")

        self.jugador_y = FILAS - 1 
        self.jugador_x = self.cfg.columnas // 2 
        self.generar_obstaculos()
        
        '''self.pantalla.fill(AZUL)
        texto = self.fuente_grande.render(f"NIVEL {self.nivel}", True, BLANCO)
        rect = texto.get_rect(center=(ANCHO_VENTANA//2, ALTO_VENTANA//2))
        self.pantalla.blit(texto, rect)
        
        # Mostrar velocidad actual si es automático
        if self.cfg.avance_auto:
            vel_txt = self.fuente.render(f"Velocidad: {self.cfg.velocidad}ms", True, BLANCO)
            vel_rect = vel_txt.get_rect(center=(ANCHO_VENTANA//2, ALTO_VENTANA//2 + 50))
            self.pantalla.blit(vel_txt, vel_rect)'''

        pygame.display.flip()
        #time.sleep(1) 

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

            if self.cfg.avance_auto:
                tiempo_actual = pygame.time.get_ticks()
                tiempo_transcurrido = tiempo_actual - self.ultimo_tiempo_mov
                # Calculamos el porcentaje (0.0 a 1.0) de progreso hasta el siguiente tick
                progreso = min(1.0, tiempo_transcurrido / max(1, self.cfg.velocidad))
                
                alto_barra = 12
                y_barra = ALTO_VENTANA - 60  # Posición de la barra justo encima del texto
                
                # Fondo gris oscuro
                pygame.draw.rect(self.pantalla, GRIS, (0, y_barra, ANCHO_VENTANA, alto_barra))
                # Relleno blanco dependiente del tiempo
                pygame.draw.rect(self.pantalla, AZUL, (0, y_barra, ANCHO_VENTANA * progreso, alto_barra))

            # UI
            info_score = self.fuente.render(f"Puntos: {self.puntuacion}", True, BLANCO)
            info_level = self.fuente.render(f"Nivel: {self.nivel}", True, BLANCO)
            self.pantalla.blit(info_score, (10, ALTO_VENTANA - 40))
            self.pantalla.blit(info_level, (ANCHO_VENTANA - 100, ALTO_VENTANA - 40))

        pygame.display.flip()

    def controlAssist(self, prediction, info) -> str:
        if AIMBOT_MODE:
            if info["name"] == "ExponentialSmoothing":
                umbral = 1 - SEVERIDAD_AIMBOT
                self.heuristica_aimbot_2(prediction, info, umbral)
            
        
        else:
            self.string_to_action(prediction)


    def heuristica_aimbot_1(self, prediction, info, umbral):
         
        if info["name"] == "ExponentialSmoothing":
            
            if self.jugador_x < self.posiciones_seguras[self.jugador_y]: # Derecha segura
                if info["p_right_smooth"] >= umbral:
                    self.mover_jugador(1, 0)
                else :
                    self.string_to_action(prediction)
            elif self.jugador_x > self.posiciones_seguras[self.jugador_y]: # Izquierda segura
                if info["p_left_smooth"] >= umbral:
                    self.mover_jugador(-1, 0)
                else :
                    self.string_to_action(prediction)
            else: # rest seguro
                if info["p_right_smooth"] > SEVERIDAD_AIMBOT:
                    self.mover_jugador(1, 0)
                elif info["p_left_smooth"] >  SEVERIDAD_AIMBOT:
                    self.mover_jugador(-1, 0)
                else:
                    # Nada
                    self.mover_jugador(0, 0)

    def heuristica_aimbot_2(self, prediction, info, umbral):
        # Definición de colores ANSI
        CYAN = '\033[96m'
        VERDE = '\033[92m'
        AMARILLO = '\033[93m'
        ROJO = '\033[91m'
        RESET = '\033[0m'
        NEGRILLA = '\033[1m'

        #print(f"{NEGRILLA}--- DEBUG AIMBOT ---{RESET}")

        # si no tenemos un obstáculo encima
        #print(f"[DEBUG info] j_y = {self.jugador_y}; matriz_act = {self.matriz_obstaculos[self.jugador_x][self.jugador_y]}; matriz_sig = {self.matriz_obstaculos[self.jugador_x][self.jugador_y-1]}")
        if self.jugador_y == 0 or (not self.matriz_obstaculos[self.jugador_x][self.jugador_y-1]):
            #print(f"{CYAN}[Libre]{RESET} Sin obstáculos arriba.")
            
            if info["p_right_smooth"] > SEVERIDAD_AIMBOT:
                #print(f"{VERDE}  -> Moviendo DERECHA (Suave){RESET} p_right: {info['p_right_smooth']:.2f}")
                self.mover_jugador(1, 0)
            elif info["p_left_smooth"] > SEVERIDAD_AIMBOT:
                #print(f"{VERDE}  -> Moviendo IZQUIERDA (Suave){RESET} p_left: {info['p_left_smooth']:.2f}")
                self.mover_jugador(-1, 0)
            else:
                #print(f"{AMARILLO}  -> QUIETO (Sin tendencia clara){RESET}")
                self.mover_jugador(0, 0)

        # Derecha segura
        elif self.jugador_x < self.posiciones_seguras[self.jugador_y]:
            #print(f"{AMARILLO}[Obstáculo]{RESET} Buscando Derecha Segura...")
            
            if info["p_right_smooth"] >= umbral:
                #print(f"{VERDE}  -> Moviendo DERECHA (Supera umbral){RESET}")
                self.mover_jugador(1, 0)
            else:
                #print(f"{ROJO}  -> Siguiendo Predicción (Bajo umbral){RESET} Pred: {prediction}")
                self.string_to_action(prediction)

        # Izquierda segura
        elif self.jugador_x > self.posiciones_seguras[self.jugador_y]:
            #print(f"{AMARILLO}[Obstáculo]{RESET} Buscando Izquierda Segura...")
            
            if info["p_left_smooth"] >= umbral:
                #print(f"{VERDE}  -> Moviendo IZQUIERDA (Supera umbral){RESET}")
                self.mover_jugador(-1, 0)
            else:
                #print(f"{ROJO}  -> Siguiendo Predicción (Bajo umbral){RESET} Pred: {prediction}")
                self.string_to_action(prediction)

    def heuristica_aimbot_3(self, prediction, info, umbral):
        if self.jugador_y == 0 or (not self.matriz_obstaculos[self.jugador_x][self.jugador_y-1]):
            if info["p_right_smooth"] > SEVERIDAD_AIMBOT:
                self.mover_jugador(1, 0)
            elif info["p_left_smooth"] >  SEVERIDAD_AIMBOT:
                self.mover_jugador(-1, 0)
            else:
                self.mover_jugador(0, 0)
        elif self.jugador_x < self.posiciones_seguras[self.jugador_y]: # Derecha segura
            print(f"[DEBUG] DCHAAA")
            if info["p_right_smooth"] >= umbral:
                self.mover_jugador(1, 0)
            else :
                self.string_to_action(prediction)
        elif self.jugador_x > self.posiciones_seguras[self.jugador_y]: # Izquierda segura
            print(f"[DEBUG] IZQDAAA")
            if info["p_left_smooth"] >= umbral:
                self.mover_jugador(-1, 0)
            else :
                self.string_to_action(prediction)

    '''def heuristica_aimbot_4(self, prediction, info, umbral):
        if prediction == "rest":
            
        elif prediction == "right_hand":
        else:'''
            


    def string_to_action(self, prediction):
        if prediction == "left_hand":
            self.mover_jugador(-1, 0)
        elif prediction == "right_hand":
            self.mover_jugador(1, 0)
        else:
            self.mover_jugador(0, 0)





    def start(self, filename: Optional[str] = None) -> None:

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
                
                '''if event.type == PREDICTED_LEFT and running:
                    self.mover_jugador(-1, 0)
                
                if event.type == PREDICTED_RIGHT and running:
                    self.mover_jugador(1, 0)'''
            
            # ── Pipe BCI — no bloqueante ──────────────────────────────────────
            while self.pipe.poll(0):
                try:
                    prediction, info = self.read_msg()
                    self.controlAssist(prediction, info)
                except Exception:
                    pass

            if self.cfg.avance_auto and not self.game_over:
                ahora = pygame.time.get_ticks()
                if ahora - self.ultimo_tiempo_mov > self.cfg.velocidad:
                    self.mover_jugador(0, -1)
                    self.ultimo_tiempo_mov = ahora

            self.dibujar()
            self.reloj.tick(60)

        pygame.quit()
        sys.exit()


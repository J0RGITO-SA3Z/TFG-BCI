import pygame
import random
import time
import copy  # Necesario para clonar el estado de los tubos

# --- CONFIGURACIÓN ---
T = 6
N = 4
COLORS_LIST = [
    (255, 50, 50),   # Rojo
    (50, 255, 50),   # Verde
    (50, 50, 255),   # Azul
    (255, 255, 50),  # Amarillo
]
TUBES_PER_ROW = 4
WIDTH, HEIGHT = 800, 600
FPS = 60

class BubbleSortGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Bubble Sort Game - con sistema Undo")
        self.font = pygame.font.SysFont("Arial", 24)
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self):
        pool = []
        num_colors_needed = (T * N) // N - 1 
        self.colors = COLORS_LIST[:num_colors_needed]
        
        for c in self.colors:
            pool += [c] * N
        random.shuffle(pool)
        
        self.tubes = [[] for _ in range(T)]
        for i, ball in enumerate(pool):
            self.tubes[i % (T-1)].append(ball)
            
        self.selected = 0
        self.holding = None
        self.moves = 0
        self.start_time = time.time()
        self.game_over = False
        self.win = False
        
        # --- HISTORIAL PARA UNDO ---
        self.history = [] 

    def save_state(self):
        """Guarda una copia del estado actual antes de realizar una acción."""
        state = {
            'tubes': copy.deepcopy(self.tubes),
            'holding': self.holding,
            'moves': self.moves
        }
        self.history.append(state)
        # Limitar el historial a los últimos 50 movimientos para no saturar memoria
        if len(self.history) > 50:
            self.history.pop(0)

    def undo(self):
        """Revierte a la última acción guardada."""
        if self.history:
            last_state = self.history.pop()
            self.tubes = last_state['tubes']
            self.holding = last_state['holding']
            self.moves = last_state['moves']
            print("Acción revertida")
        else:
            print("No hay más movimientos para deshacer")

    def check_win(self):
        for t in self.tubes:
            if len(t) == 0: continue
            if len(t) < N or not all(c == t[0] for c in t):
                return False
        return True

    def can_move(self):
        if self.holding: return True
        for i, src in enumerate(self.tubes):
            if not src: continue
            ball = src[-1]
            for j, dst in enumerate(self.tubes):
                if i == j: continue
                # Es posible mover si el destino tiene sitio Y (está vacío o tiene el mismo color arriba)
                if len(dst) < N and (not dst or dst[-1] == ball):
                    return True
        return False

    def draw(self):
        self.screen.fill((30, 30, 30))
        
        timer = int(time.time() - self.start_time) if not self.game_over else self.final_time
        txt = self.font.render(f"Movimientos: {self.moves}  Tiempo: {timer}s  (Z para deshacer)", True, (255, 255, 255))
        self.screen.blit(txt, (20, 20))

        margin_x, margin_y = 100, 100
        spacing_x, spacing_y = 150, 180

        for i in range(T):
            row, col = i // TUBES_PER_ROW, i % TUBES_PER_ROW
            x, y = margin_x + col * spacing_x, margin_y + row * spacing_y
            pygame.draw.rect(self.screen, (200, 200, 200), (x, y, 60, N * 40 + 10), 3)
            
            if i == self.selected:
                pygame.draw.polygon(self.screen, (255, 255, 255), [(x+20, y-10), (x+40, y-10), (x+30, y-2)])
                if self.holding:
                    pygame.draw.circle(self.screen, self.holding, (x + 30, y - 30), 15)

            for j, ball_color in enumerate(self.tubes[i]):
                bx, by = x + 30, (y + (N * 40)) - (j * 40) - 20
                pygame.draw.circle(self.screen, ball_color, (bx, by), 15)

        if self.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg_text = "¡GANASTE!"
                msg_color = (100, 255, 100)
            else:
                msg_text = "Has perdido :( no te quedan movimientos disponibles"
                msg_color = (255, 100, 100)
            
            t1 = self.font.render(msg_text, True, msg_color)
            t2 = self.font.render("Pulsa R para reiniciar o Q para salir", True, (255, 255, 255))
            
            # Centrar textos
            self.screen.blit(t1, (WIDTH // 2 - t1.get_width() // 2, HEIGHT // 2 - 40))
            self.screen.blit(t2, (WIDTH // 2 - t2.get_width() // 2, HEIGHT // 2 + 10))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if self.game_over:
                        if event.key == pygame.K_r: self.reset_game()
                        if event.key == pygame.K_q: running = False
                        continue

                    # --- NUEVO CONTROL: UNDO ---
                    if event.key == pygame.K_z:
                        self.undo()

                    if event.key == pygame.K_d:
                        self.selected = (self.selected + 1) % T
                    
                    if event.key == pygame.K_a:
                        if self.holding is None:
                            if self.tubes[self.selected]:
                                self.save_state() # Guardamos antes de coger
                                self.holding = self.tubes[self.selected].pop()
                        else:
                            target = self.tubes[self.selected]
                            # Regla: Tubo vacío o mismo color
                            if not target or (len(target) < N and target[-1] == self.holding):
                                self.save_state() # Guardamos antes de soltar
                                target.append(self.holding)
                                self.holding = None
                                self.moves += 1
                                
                                # Comprobar condiciones de fin
                                if self.check_win():
                                    self.win = True
                                    self.game_over = True
                                    self.final_time = int(time.time() - self.start_time)
                                elif not self.can_move():
                                    self.win = False
                                    self.game_over = True
                                    self.final_time = int(time.time() - self.start_time)
                            else:
                                print("Movimiento no permitido")

            self.clock.tick(FPS)
        pygame.quit()

if __name__ == "__main__":
    BubbleSortGame().run()
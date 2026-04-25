from __future__ import annotations

import random
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from typing import Optional

import pygame

from EEG_app.src.components.RT_pipe_components.MIGames.MIGame import MIGame

# ── Colores ──────────────────────────────────────────────────────────────────
WHITE    = (255, 255, 255)
BLACK    = (0,   0,   0  )
YELLOW   = (255, 204, 0  )
GREEN    = (34,  177, 76 )
SKY_BLUE = (113, 197, 207)
RED      = (220, 50,  50 )

# ── Dimensiones ───────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 900, 600
PIPE_WIDTH    = 70
PIPE_GAP      = 160
PIPE_SPEED    = 1
GRAVITY       = 0.1
JUMP_STRENGTH = -3
BIRD_X        = 80
BIRD_RADIUS   = 15
SPAWN_INTERVAL_MS = 5000
FPS           = 60


class FlappyBirdMIGame(MIGame):
    """Flappy Bird controlado por predicciones MI recibidas por pipe."""

    def __init__(self, pipe: Connection, stop_event: Event) -> None:
        super().__init__(pipe, stop_event)

    # ── Helpers de dibujo ────────────────────────────────────────────────────

    def _draw(self, screen: pygame.Surface, font: pygame.font.Font,
              bird_y: float, pipes: list, score: int,
              game_active: bool, bci_flash: int) -> None:
        screen.fill(SKY_BLUE)

        for pipe in pipes:
            pygame.draw.rect(screen, GREEN, (pipe['x'], 0, PIPE_WIDTH, pipe['top']))
            pygame.draw.rect(screen, BLACK, (pipe['x'], 0, PIPE_WIDTH, pipe['top']), 2)
            bottom_y = pipe['top'] + PIPE_GAP
            pygame.draw.rect(screen, GREEN, (pipe['x'], bottom_y, PIPE_WIDTH, HEIGHT - bottom_y))
            pygame.draw.rect(screen, BLACK, (pipe['x'], bottom_y, PIPE_WIDTH, HEIGHT - bottom_y), 2)

        # Flash rojo alrededor del pájaro cuando llega predicción
        if bci_flash > 0:
            pygame.draw.circle(screen, RED, (BIRD_X, int(bird_y)), BIRD_RADIUS + 5)

        pygame.draw.circle(screen, YELLOW, (BIRD_X, int(bird_y)), BIRD_RADIUS)
        pygame.draw.circle(screen, BLACK,  (BIRD_X, int(bird_y)), BIRD_RADIUS, 2)

        score_surf = font.render(str(score), True, WHITE)
        screen.blit(score_surf, (WIDTH // 2 - score_surf.get_width() // 2, 50))

        if not game_active:
            over   = font.render("¡CHOCASTE!", True, BLACK)
            restart = font.render("Pulsa ESPACIO para reiniciar", True, WHITE)
            screen.blit(over,    (WIDTH // 2 - over.get_width()    // 2, HEIGHT // 2 - 40))
            screen.blit(restart, (WIDTH // 2 - restart.get_width() // 2, HEIGHT // 2 + 10))

        pygame.display.update()

    @staticmethod
    def _check_collisions(bird_y: float, pipes: list) -> bool:
        if bird_y <= BIRD_RADIUS or bird_y >= HEIGHT - BIRD_RADIUS:
            return False
        bird_rect = pygame.Rect(BIRD_X - BIRD_RADIUS, bird_y - BIRD_RADIUS,
                                BIRD_RADIUS * 2, BIRD_RADIUS * 2)
        for pipe in pipes:
            top_rect    = pygame.Rect(pipe['x'], 0, PIPE_WIDTH, pipe['top'])
            bottom_y    = pipe['top'] + PIPE_GAP
            bottom_rect = pygame.Rect(pipe['x'], bottom_y, PIPE_WIDTH, HEIGHT - bottom_y)
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                return False
        return True

    @staticmethod
    def _reset_state() -> tuple:
        """Devuelve el estado inicial del juego."""
        return HEIGHT // 2, 0.0, [], 0, True   # bird_y, vel, pipes, score, active

    # ── Interfaz MIGame ───────────────────────────────────────────────────────

    def start(self, filename: Optional[str] = None) -> None:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Flappy Bird – BCI MI Game")
        clock = pygame.time.Clock()
        font  = pygame.font.SysFont(None, 48)

        SPAWN_PIPE = pygame.USEREVENT
        pygame.time.set_timer(SPAWN_PIPE, SPAWN_INTERVAL_MS)

        bird_y, bird_vel, pipes, score, game_active = self._reset_state()
        bci_flash = 0   # fotogramas que dura el flash visual de predicción BCI

        while not self.stop_event.is_set():

            # ── 1. Eventos pygame ────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if game_active:
                        bird_vel = JUMP_STRENGTH
                    else:
                        bird_y, bird_vel, pipes, score, game_active = self._reset_state()

                if event.type == SPAWN_PIPE and game_active:
                    top = random.randint(50, HEIGHT - PIPE_GAP - 50)
                    pipes.append({'x': WIDTH, 'top': top, 'passed': False})

            # ── 2. Pipe BCI — no bloqueante, igual que una tecla ─────────────
            # poll(0) devuelve True si hay datos listos; nunca bloquea.
            while self.pipe.poll(0):
                try:
                    prediction, info = self.read_msg()
                    action = self.controlAssist(prediction, info)

                    if action == "jump":
                        if game_active:
                            bird_vel = JUMP_STRENGTH
                        else:
                            bird_y, bird_vel, pipes, score, game_active = self._reset_state()
                    bci_flash = 12  # mantener flash ~0.2 s a 60 FPS
                except Exception:
                    pass  # mensaje malformado; ignorar

            # ── 3. Física del juego ───────────────────────────────────────────
            if game_active:
                bird_vel += GRAVITY
                bird_y   += bird_vel

                for pipe in pipes:
                    pipe['x'] -= PIPE_SPEED
                    if pipe['x'] + PIPE_WIDTH < BIRD_X and not pipe['passed']:
                        score += 1
                        pipe['passed'] = True

                pipes = [p for p in pipes if p['x'] + PIPE_WIDTH > 0]
                game_active = self._check_collisions(bird_y, pipes)

            if bci_flash > 0:
                bci_flash -= 1

            # ── 4. Render ────────────────────────────────────────────────────
            self._draw(screen, font, bird_y, pipes, score, game_active, bci_flash)
            clock.tick(FPS)

        pygame.quit()
    
    def controlAssist(self, prediction, info):
        """Lógica de control: recibe la predicción final y el dict de info del filtro."""
        if prediction == "right_hand":
            return "jump"
        return "none"

import pygame
import random
import sys

# 1. Configuración inicial
pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird - BCI Test")
clock = pygame.time.Clock()

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 204, 0)
GREEN = (34, 177, 76)
SKY_BLUE = (113, 197, 207)

# 2. Variables del pájaro
bird_x = 80  # A la izquierda, pero no pegado al borde
bird_y = HEIGHT // 2
bird_radius = 15
bird_velocity = 0
gravity = 0.1
jump_strength = -3

# 3. Variables de las tuberías
pipe_width = 70
pipe_gap = 160  # Hueco entre la tubería de arriba y la de abajo
pipe_speed = 2
pipes = []
next_pipe_time = 2500

# Evento para generar tuberías nuevas cada 1.5 segundos
SPAWN_PIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWN_PIPE, next_pipe_time)

font = pygame.font.SysFont(None, 48)
score = 0
game_active = True

def draw_game():
    screen.fill(SKY_BLUE)
    
    # Dibujar tuberías
    for pipe in pipes:
        # Tubería superior
        pygame.draw.rect(screen, GREEN, (pipe['x'], 0, pipe_width, pipe['top_height']))
        pygame.draw.rect(screen, BLACK, (pipe['x'], 0, pipe_width, pipe['top_height']), 2)
        # Tubería inferior
        bottom_y = pipe['top_height'] + pipe_gap
        bottom_height = HEIGHT - bottom_y
        pygame.draw.rect(screen, GREEN, (pipe['x'], bottom_y, pipe_width, bottom_height))
        pygame.draw.rect(screen, BLACK, (pipe['x'], bottom_y, pipe_width, bottom_height), 2)
        
    # Dibujar pájaro
    pygame.draw.circle(screen, YELLOW, (int(bird_x), int(bird_y)), bird_radius)
    pygame.draw.circle(screen, BLACK, (int(bird_x), int(bird_y)), bird_radius, 2)
    
    # Dibujar puntuación
    score_text = font.render(str(score), True, WHITE)
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 50))
    
    # Pantalla de Game Over
    if not game_active:
        over_text = font.render("¡CHOCASTE!", True, BLACK)
        restart_text = font.render("Pulsa ESPACIO", True, WHITE)
        screen.blit(over_text, (WIDTH // 2 - over_text.get_width() // 2, HEIGHT // 2 - 40))
        screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 10))

    pygame.display.update()

def check_collisions():
    # Chocar contra el techo o el suelo
    if bird_y <= bird_radius or bird_y >= HEIGHT - bird_radius:
        return False
        
    # Chocar contra tuberías
    bird_rect = pygame.Rect(bird_x - bird_radius, bird_y - bird_radius, bird_radius * 2, bird_radius * 2)
    for pipe in pipes:
        top_rect = pygame.Rect(pipe['x'], 0, pipe_width, pipe['top_height'])
        bottom_y = pipe['top_height'] + pipe_gap
        bottom_rect = pygame.Rect(pipe['x'], bottom_y, pipe_width, HEIGHT - bottom_y)
        
        if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
            return False
            
    return True

# =========================================================
# BUCLE PRINCIPAL DEL JUEGO
# =========================================================
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
        # Controles
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # AQUÍ ES DONDE ENTRARÁ TU PREDICCIÓN DE LA BCI EN EL FUTURO
                if game_active:
                    bird_velocity = jump_strength
                else:
                    # Reiniciar juego
                    game_active = True
                    pipes.clear()
                    bird_y = HEIGHT // 2
                    bird_velocity = 0
                    score = 0
                    
        # Generar tuberías
        if event.type == SPAWN_PIPE and game_active:
            min_height = 50
            max_height = HEIGHT - pipe_gap - 50
            top_height = random.randint(min_height, max_height)
            pipes.append({'x': WIDTH, 'top_height': top_height, 'passed': False})

    if game_active:
        # Físicas del pájaro
        bird_velocity += gravity
        bird_y += bird_velocity
        
        # Mover tuberías
        for pipe in pipes:
            pipe['x'] -= pipe_speed
            # Sumar puntos cuando pasas la tubería
            if pipe['x'] + pipe_width < bird_x and not pipe['passed']:
                score += 1
                pipe['passed'] = True
                
        # Limpiar memoria (borrar tuberías que ya salieron de la pantalla por la izquierda)
        pipes = [pipe for pipe in pipes if pipe['x'] + pipe_width > 0]
        
        # Comprobar si morimos
        game_active = check_collisions()

    draw_game()
    clock.tick(60) # Mantener a 60 FPS
import pygame
import sys

# --- CONFIGURACIÓN ---
ANCHO, ALTO = 800, 600
COLOR_FONDO = (30, 30, 30)       # Gris oscuro
COLOR_NEUTRO = (200, 200, 200)   # Blanco/Gris claro (Círculo)
COLOR_ACTIVO = (0, 255, 0)       # Verde (Flechas)
RADIO_CIRCULO = 80
TAMANO_FLECHA = 80

def dibujar_flecha(surface, direccion, color, centro_x, centro_y):
    """
    Dibuja un triángulo apuntando a la dirección deseada.
    """
    if direccion == "arriba": # W
        puntos = [
            (centro_x, centro_y - TAMANO_FLECHA),          # Punta arriba
            (centro_x - TAMANO_FLECHA, centro_y + TAMANO_FLECHA // 2), # Abajo Izq
            (centro_x + TAMANO_FLECHA, centro_y + TAMANO_FLECHA // 2)  # Abajo Der
        ]
    elif direccion == "izquierda": # A
        puntos = [
            (centro_x - TAMANO_FLECHA, centro_y),          # Punta Izquierda
            (centro_x + TAMANO_FLECHA // 2, centro_y - TAMANO_FLECHA), # Arriba Der
            (centro_x + TAMANO_FLECHA // 2, centro_y + TAMANO_FLECHA)  # Abajo Der
        ]
    elif direccion == "derecha": # D
        puntos = [
            (centro_x + TAMANO_FLECHA, centro_y),          # Punta Derecha
            (centro_x - TAMANO_FLECHA // 2, centro_y - TAMANO_FLECHA), # Arriba Izq
            (centro_x - TAMANO_FLECHA // 2, centro_y + TAMANO_FLECHA)  # Abajo Izq
        ]
    
    pygame.draw.polygon(surface, color, puntos)

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Test BCI - Feedback Visual")
    reloj = pygame.time.Clock()

    centro_x, centro_y = ANCHO // 2, ALTO // 2

    while True:
        # 1. Gestionar cierre de ventana
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 2. Leer estado del teclado (Real o Simulado por pydirectinput)
        teclas = pygame.key.get_pressed()
        
        # Limpiar pantalla
        pantalla.fill(COLOR_FONDO)

        # 3. Lógica de dibujo
        if teclas[pygame.K_w]:
            # Tecla W -> Flecha Arriba (Feet/Pies)
            dibujar_flecha(pantalla, "arriba", COLOR_ACTIVO, centro_x, centro_y)
            texto = "AVANZAR / PIES"
            
        elif teclas[pygame.K_a]:
            # Tecla A -> Flecha Izquierda
            dibujar_flecha(pantalla, "izquierda", COLOR_ACTIVO, centro_x, centro_y)
            texto = "IZQUIERDA"
            
        elif teclas[pygame.K_d]:
            # Tecla D -> Flecha Derecha
            dibujar_flecha(pantalla, "derecha", COLOR_ACTIVO, centro_x, centro_y)
            texto = "DERECHA"
            
        else:
            # Ninguna tecla -> Círculo (Descanso)
            pygame.draw.circle(pantalla, COLOR_NEUTRO, (centro_x, centro_y), RADIO_CIRCULO)
            texto = "REPOSO"

        # (Opcional) Dibujar etiqueta de texto abajo
        fuente = pygame.font.SysFont("Arial", 30)
        superficie_texto = fuente.render(texto, True, (255, 255, 255))
        rect_texto = superficie_texto.get_rect(center=(centro_x, ALTO - 50))
        pantalla.blit(superficie_texto, rect_texto)

        # Actualizar pantalla
        pygame.display.flip()
        
        # 60 FPS para que sea fluido
        reloj.tick(60)

if __name__ == "__main__":
    main()
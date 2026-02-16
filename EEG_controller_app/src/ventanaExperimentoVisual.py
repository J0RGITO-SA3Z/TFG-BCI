import pygame
import multiprocessing as mp

class ventanaExperimentoVisual:
    def __init__(self):
        self.cola = None
        self.child = None

    def open(self):
        self.cola = mp.Queue()
        self.child = mp.Process(target=window_process,args= (self.cola,))
        self.child.start()

    def draw_text(self, text):
        if self.cola != None:
            self.cola.put(text)

    def close(self):
        if self.cola != None:
            self.cola.put("STOP")


def window_process(queue):

    fullscreen = True
    font_size = 120
    screen = None
    font = None
    width = None
    height = None

    pygame.init()
    info = pygame.display.Info()

    width = info.current_w
    height = info.current_h

    if fullscreen:
        screen = pygame.display.set_mode(
            (width, height),
            pygame.FULLSCREEN
        )
    else:
        screen = pygame.display.set_mode((1280, 720))

    font = pygame.font.SysFont("Arial", font_size)
    
    reception = queue.get()
    
    while reception != "STOP":
        draw_text(screen, width, height,font, reception, (255,255,255),True)
        reception = queue.get()
        
    pygame.quit()

    return



def draw_text(screen, width, height, font, text, color=(255, 255, 255), clear=True):

    if clear:
        screen.fill((0, 0, 0))

    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(width // 2, height // 2))
    screen.blit(surf, rect)
    pygame.display.flip()
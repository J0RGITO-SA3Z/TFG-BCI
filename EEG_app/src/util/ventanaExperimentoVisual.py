import pygame
import multiprocessing as mp

class ventanaExperimentoVisual:
    def __init__(self):
        self.cola = None
        self.child = None

    def open(self):
        self.cola = mp.Queue()
        self.child = mp.Process(target=window_process, args=(self.cola,))
        self.child.start()

    def draw_text(self, text, current=None, total=None):
        if self.cola is not None:
            counter = f"{current} / {total}" if current is not None and total is not None else None
            self.cola.put({"main": text, "counter": counter})

    def close(self):
        if self.cola is not None:
            self.cola.put("STOP")


def window_process(queue):
    pygame.init()
    info = pygame.display.Info()
    width = info.current_w
    height = info.current_h

    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)

    font_main = pygame.font.SysFont("Arial", 120)
    font_counter = pygame.font.SysFont("Arial", 28)

    msg = queue.get()

    while msg != "STOP":
        screen.fill((0, 0, 0))

        if isinstance(msg, dict):
            main_text = msg.get("main", "")
            counter_text = msg.get("counter")
        else:
            main_text = msg
            counter_text = None

        if main_text:
            surf = font_main.render(main_text, True, (255, 255, 255))
            rect = surf.get_rect(center=(width // 2, height // 2))
            screen.blit(surf, rect)

        if counter_text:
            c_surf = font_counter.render(counter_text, True, (160, 160, 160))
            c_rect = c_surf.get_rect(bottomright=(width - 20, height - 20))
            screen.blit(c_surf, c_rect)

        pygame.display.flip()
        msg = queue.get()

    pygame.quit()

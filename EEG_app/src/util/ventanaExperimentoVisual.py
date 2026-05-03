import pygame
import multiprocessing as mp
import queue as _queue


class ventanaExperimentoVisual:
    def __init__(self):
        self.cola = None
        self.child = None

    def open(self):
        self.cola = mp.Queue()
        self.child = mp.Process(target=window_process, args=(self.cola,), daemon=True)
        self.child.start()

    def draw_text(self, text, current=None, total=None):
        if self.cola is not None:
            counter = f"{current} / {total}" if current is not None and total is not None else None
            self.cola.put({"main": text, "counter": counter})

    def close(self):
        if self.cola is not None:
            self.cola.put("STOP")
        if self.child is not None:
            self.child.join(timeout=3)
            if self.child.is_alive():
                self.child.terminate()


def window_process(queue):
    pygame.init()
    info = pygame.display.Info()
    width, height = info.current_w, info.current_h

    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    font_main = pygame.font.SysFont("Arial", 120)
    font_counter = pygame.font.SysFont("Arial", 28)
    clock = pygame.time.Clock()

    main_text = ""
    counter_text = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        try:
            msg = queue.get_nowait()
            if msg == "STOP":
                running = False
            elif isinstance(msg, dict):
                main_text = msg.get("main", "")
                counter_text = msg.get("counter")
            else:
                main_text = str(msg)
                counter_text = None
        except _queue.Empty:
            pass

        screen.fill((0, 0, 0))

        if main_text:
            surf = font_main.render(main_text, True, (255, 255, 255))
            rect = surf.get_rect(center=(width // 2, height // 2))
            screen.blit(surf, rect)

        if counter_text:
            c_surf = font_counter.render(counter_text, True, (160, 160, 160))
            c_rect = c_surf.get_rect(bottomright=(width - 20, height - 20))
            screen.blit(c_surf, c_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

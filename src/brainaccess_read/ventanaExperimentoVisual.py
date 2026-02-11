import pygame

class ventanaExperimentoVisual:
    def __init__(self, fullscreen=True, font_size=80):
        self.fullscreen = fullscreen
        self.font_size = font_size
        self.screen = None
        self.font = None
        self.width = None
        self.height = None

    def open(self):
        pygame.init()
        info = pygame.display.Info()

        self.width = info.current_w
        self.height = info.current_h

        if self.fullscreen:
            self.screen = pygame.display.set_mode(
                (self.width, self.height),
                pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode((1280, 720))

        self.font = pygame.font.SysFont("Arial", self.font_size)

    def draw_text(self, text, color=(255, 255, 255), clear=True):
        if clear:
            self.screen.fill((0, 0, 0))

        surf = self.font.render(text, True, color)
        rect = surf.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(surf, rect)
        pygame.display.flip()

    def close(self):
        pygame.quit()
import pygame
import math
import numpy as np

from gym_env.envs.car import Car

# 870 X 746
screen_width = 870 # 1500 
screen_height = 746 # 800
w_ini = screen_width//2 - 320
h_ini = screen_height//2 + 300 #+ 300


class PyGame2D:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 15)
        self.car = Car('assets/terrasentia.png', 'assets/map_new7_small.png', [w_ini, h_ini])
        # self.car = Car('assets/terrasentia.png', 'assets/map2_small.png', [w_ini, h_ini])
        self.game_speed = 60
        self.mode = 0

        pygame.draw.circle(self.screen, (255, 0, 0), (0, 0), 100)

    # speed = v |(action 0)
    # angle = w |(action 1)
    def action(self, action):
        action0 = np.select([action[0] == 0, action[0] == 2], [1, 9], default=5)
        action1 = np.select([action[1] == 0, action[1] == 2], [-5, 5], default=0)

        self.car.speed = action0
        self.car.w     = action1

        self.car.update()
        self.car.check_collision()

        self.car.radars.clear()
        for d in range(-45, 46, 45):
            self.car.check_radar(d)

    def evaluate(self):
        reward = 0
        w1 = 1 # negatie 
        w2 = 0.1 # positive
        if not self.car.is_alive: 
            reward = -5 #- w1*self.car.speed
        else: 
            reward = +0.5 #+ w2*self.car.speed
        #print(f'reward: {reward:.2f}, speed: {self.car.speed}, distance: {self.car.distance}')
        return reward

    def is_done(self):
        if not self.car.is_alive:
            self.car.current_check = 0
            self.car.distance = 0
            return True
        return False

    def observe(self):
        # return state
        radars = self.car.radars
        ret = [0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)
        return tuple(ret)

    def view(self):
        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode += 1
                    self.mode = self.mode % 3

        self.screen.blit(self.car.map, (0, 0))

        if self.mode == 1:
            self.screen.fill((0, 0, 0))

        self.car.radars_for_draw.clear()
        for d in range(-45, 46, 45):
            self.car.check_radar_for_draw(d)

        #pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        self.car.draw_collision(self.screen)
        self.car.draw_radar(self.screen)
        self.car.draw(self.screen)

        pygame.display.flip()
        self.clock.tick(self.game_speed)




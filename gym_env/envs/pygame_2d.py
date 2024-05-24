import pygame
import math

screen_width = 960 # 1500 
screen_height = 540 # 800
w_ini = screen_height//2 + 150
h_ini = screen_width//2 - 35
#TODO: ADD CHECKPOINTS
check_point = ((1200, 660), (1250, 120), (190, 200), (1030, 270), (250, 475), (650, 690))

class Car:
    def __init__(self, car_file, map_file, pos):
        self.surface = pygame.image.load(car_file)
        self.map = pygame.image.load(map_file)
        self.surface = pygame.transform.scale(self.surface, (60, 60))
        self.rotate_surface = self.surface
        self.pos = pos
        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 30, self.pos[1] + 30]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.current_check = 0
        self.prev_distance = 0
        self.cur_distance = 0
        self.goal = False
        self.check_flag = False
        self.distance = 0
        self.time_spent = 0
        for d in range(-45, 46, 45):
            self.check_radar(d)

        for d in range(-45, 46, 45):
            self.check_radar_for_draw(d)


    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)

    def draw_collision(self, screen):
        for i in range(4):
            x = int(self.four_points[i][0])
            y = int(self.four_points[i][1])
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 5)

    def draw_radar(self, screen):
        for r in self.radars_for_draw:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self):
        self.is_alive = True
        for p in self.four_points:
            if self.map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.map.get_at((x, y)) == (255, 255, 255, 255) and len < 300:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])


    def check_radar_for_draw(self, degree):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.map.get_at((x, y)) == (255, 255, 255, 255) and len < 300:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars_for_draw.append([(x, y), dist])

    def update(self):
        #check speed
        self.speed -= 1.0
        if self.speed > 10:
            self.speed = 10
        if self.speed < 1:
            self.speed = 1

        #check position
        self.rotate_surface = rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120
        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

        # caculate 4 collision points
        self.center = [int(self.pos[0]) + 30, int(self.pos[1]) + 30]
        len = 20
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

class PyGame2D:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 15)
        self.car = Car('assets/terrasentia.png', 'assets/map_grey_small.png', [w_ini, h_ini])
        # self.car = Car('assets/terrasentia.png', 'assets/map2_small.png', [w_ini, h_ini])
        self.game_speed = 60
        self.mode = 0

    def action(self, action):
        if action == 0:
            self.car.speed += 1.5
        if action == 1:
            self.car.angle += 5
        elif action == 2:
            self.car.angle -= 5

        self.car.update()
        self.car.check_collision()

        self.car.radars.clear()
        for d in range(-45, 46, 45):
            self.car.check_radar(d)

    def evaluate(self):
        reward = 0
        w1 = 0.7 # negatie 
        w2 = 0.1 # positive
        if not self.car.is_alive: 
            reward = -10 - w1*self.car.speed
        else: 
            reward = +0.5 + w2*self.car.speed
        #print(f'reward: {reward:.2f}, speed: {self.car.speed}, distance: {self.car.distance}')
        return reward

    def is_done(self):
        if not self.car.is_alive or self.car.goal:
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

        pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        self.car.draw_collision(self.screen)
        self.car.draw_radar(self.screen)
        self.car.draw(self.screen)

        # text = self.font.render("Press 'm' to change view mode", True, (255, 0, 0))
        # text_rect = text.get_rect()
        # text_rect.center = (screen_width/2, 150)
        # self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(self.game_speed)


def rot_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image

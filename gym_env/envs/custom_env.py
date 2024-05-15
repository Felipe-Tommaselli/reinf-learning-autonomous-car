import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_env.envs.pygame_2d import PyGame2D

class CustomEnv(gym.Env):
    #metadata = {'render.modes' : ['human']}
    def __init__(self):
        self.pygame = PyGame2D()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=np.int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        del self.pygame
        self.pygame = PyGame2D()
        obs = self.pygame.observe()
        info = {}
        return obs, info

    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, False, {}

    def render(self, mode="human", close=False):
        self.pygame.view()

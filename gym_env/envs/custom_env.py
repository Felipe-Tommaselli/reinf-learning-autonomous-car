import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_env.envs.pygame_2d import PyGame2D

class CustomEnv(gym.Env):
    #metadata = {'render.modes' : ['human']}
    def __init__(self):
        self.pygame = PyGame2D()
        #self.action_space = spaces.Discrete(3)
        # action = [speed, angle]
        self.action_space            = spaces.Box(low=np.array([0, -45]),
                                            high=np.array([10, 45]),
                                            dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), 
                                            high=np.array([10, 10, 10]), 
                                            dtype=int)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        del self.pygame
        self.pygame = PyGame2D()
        obs = self.pygame.observe()
        # Return only the observation unless return_info is True
        if options and options.get('return_info', False):
            return (obs,)
        else:
            info = {}
            return (obs, info)
            
    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, False, {}

    def render(self, mode="human", close=False):
        self.pygame.view()

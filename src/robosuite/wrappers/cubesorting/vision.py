import copy
import gymnasium as gym
import robosuite as suite
import numpy as np
import cv2
import os

class CubeSortingVisionWrapper(gym.Wrapper):
    def __init__(self, env, image_size=256):
        # Run super method
        super().__init__(env=env)
        self.env = env
        # specify the observation space dtype for the vision wrapper
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8)
        self.action_space = self.env.action_space

    def set_task(self, task):
        obj_to_pick, place_to_drop = task
        self.env.unwrapped.obj_to_pick = obj_to_pick
        self.env.unwrapped.place_to_drop = place_to_drop

    def reset(self, seed=None):
        try:
            obs, info = self.env.reset()
        except:
            obs = self.env.reset()
            info = {}
        return obs, info
    
    def step(self, action):
        truncated = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
import copy
import gymnasium as gym
import robosuite as suite
import numpy as np
import cv2
import os

class HanoiVisionWrapper(gym.Wrapper):
    def __init__(self, env):
        # Run super method
        super().__init__(env=env)
        self.env = env
        # specify the observation space dtype for the vision wrapper
        target_size = 128
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(target_size, target_size, 3), dtype=np.uint8)
        self.action_space = self.env.action_space
        # Load objects images
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'assets', 'textures')
        self.objects_image = {
            "cube1": cv2.imread(os.path.join(base_path, "number1.png")),
            "cube2": cv2.imread(os.path.join(base_path, "number2.png")),
            "cube3": cv2.imread(os.path.join(base_path, "number3.png"))
        }
        # Reshape the images to 16x16 and rotate them 90 degrees
        for k in self.objects_image:
            self.objects_image[k] = cv2.resize(self.objects_image[k], (16, 16))
            self.objects_image[k] = cv2.rotate(self.objects_image[k], cv2.ROTATE_90_CLOCKWISE)
        # Create black, grey and white images for "peg1", "peg2" and "peg3"
        self.objects_image["peg1"] = np.zeros((16, 16, 3), dtype=np.uint8)
        self.objects_image["peg2"] = np.full((16, 16, 3), 128, dtype=np.uint8)
        self.objects_image["peg3"] = np.full((16, 16, 3), 255, dtype=np.uint8)

    def patch_task_image(self, obs):
        image = cv2.flip(obs.reshape(128, 128, 3), 0)
        # Change the image to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Patch the object to pick image of cube1 on the top left corner
        if self.env.obj_to_pick in self.objects_image:
            image[0:16, 0:16] = self.objects_image[self.env.obj_to_pick]
        # Patch the object to place image of cube2 on the top right corner
        if self.env.place_to_drop is not None:
            if self.env.place_to_drop in self.objects_image:
                image[0:16, 128-16:128] = self.objects_image[self.env.place_to_drop]
        # Flatten the image
        obs = image
        return obs

    def set_task(self, obj_to_pick, place_to_drop):
        self.env.obj_to_pick = obj_to_pick
        self.env.place_to_drop = place_to_drop

    def reset(self, seed=None):
        try:
            obs, info = self.env.reset()
        except:
            obs = self.env.reset()
            info = {}
        obs = self.patch_task_image(obs)
        return obs, info
    
    def step(self, action):
        truncated = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        obs = self.patch_task_image(obs)
        return obs, reward, terminated, truncated, info
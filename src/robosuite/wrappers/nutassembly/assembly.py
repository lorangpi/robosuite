import copy
#import gymnasium as gym
import gym
import robosuite as suite
import numpy as np
from robosuite.utils.detector import NutAssemblyDetector


class AssemblyWrapper(gym.Wrapper):
    def __init__(self, env, horizon=500, render=False):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.detector = NutAssemblyDetector(self)
        self.step_count = 0
        self.horizon = horizon
        self.render_step = render

        # set up spaces
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(low=self.env.action_space.low, high=self.env.action_space.high, dtype=np.float64)
        print("Action space: ", self.action_space)

    def reset(self, seed=None):
        # Reset the environment for the drop trask
        self.step_count = 0
        try:
            obs, info = self.env.reset()
        except:
            obs = self.env.reset()
            info = {}    
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        info["state"] = state
        return obs, info

    def map_gripper(self, action):
        # Change last value of the action (called gripper_action) to a mapped discretized value of the gripper opening
        # -0.5 < gripper_action < 0.5 is mapped to 0
        # gripper_action <= -0.5 is mapped to 0.1
        # gripper_action >= -0.5 is mapped to -0.1
        # Returns the modified action array
        action_gripper = action[-1]
        if -0.5 < action_gripper < 0.5:
            action_gripper = np.array([0])
        if action_gripper <= -0.5:
            action_gripper = np.array([0.1])
        elif action_gripper >= 0.5:
            action_gripper = np.array([-0.1])
        action = np.concatenate([action[:3], action_gripper])
        return action

    def step(self, action):
        truncated = False
        action = self.map_gripper(action)
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        success = state[f"on(roundnut,roundpeg)"] and state[f"on(squarenut,squarepeg)"]
        info['is_success'] = success
        truncated = truncated or self.env.done
        terminated = (terminated or success)
        self.step_count += 1
        if self.step_count > self.horizon:
            print("Horizon reached within environment")
            terminated = True
        info["state"] = state
        if self.render_step:
            self.env.render()
        return obs, reward, terminated, truncated, info
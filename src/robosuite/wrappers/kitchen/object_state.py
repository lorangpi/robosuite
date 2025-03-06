import copy
import gymnasium as gym
import robosuite as suite
import numpy as np
import os

class KitchenStateWrapper(gym.Wrapper):
    def __init__(self, env):
        # Run super method
        super().__init__(env=env)
        self.env = env
        # set up observation space
        self.obs_dim = 10

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)
        self.action_space = self.env.action_space
        
        # Object bodies
        self.pot_body = self.env.sim.model.body_name2id('PotObject_root')
        self.bread_body = self.env.sim.model.body_name2id('cube_bread_main')
        self.serving_body = self.env.sim.model.body_name2id('ServingRegionRed_main')
        self.stove_body = self.env.sim.model.body_name2id('Stove1_main')
        self.button_body = self.env.sim.model.body_name2id('Button1_switch')
        self.obj_mapping = {'pot': self.pot_body, 'bread': self.bread_body, 'serving': self.serving_body, 'stove': self.stove_body, 'button': self.button_body}

    def get_obs(self):
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body][:3])
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        goal_pick = self.env.obj_to_pick
        if goal_pick == None:
            goal_pick = 'button'
        obj_to_pick_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[goal_pick]][:3])
        goal_drop = self.env.place_to_drop
        if goal_drop == None:
            goal_drop = 'button'
        place_to_drop_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[goal_drop]][:3])
        if 'pot' in goal_pick:
            obj_to_pick_pos = obj_to_pick_pos + np.array([0, -0.09, 0])
        obs = np.concatenate([gripper_pos, [aperture], place_to_drop_pos, obj_to_pick_pos])
        return obs

    def reset(self, seed=None):
        try:
            obs, info = self.env.reset()
        except:
            obs = self.env.reset()
            info = {}
        obs = self.get_obs()
        return obs, info
    
    def step(self, action):
        truncated = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        obs = self.get_obs()
        return obs, reward, terminated, truncated, info
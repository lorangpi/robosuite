import copy
import gymnasium as gym
import robosuite as suite
import numpy as np
import os

class AssembleStateWrapper(gym.Wrapper):
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
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.roundnut_body = self.env.sim.model.body_name2id('RoundNut_main')
        self.squarenut_body = self.env.sim.model.body_name2id('SquareNut_main')
        self.roundpeg_body = self.env.sim.model.body_name2id('peg2')
        self.squarepeg_body = self.env.sim.model.body_name2id('peg1')
        self.obj_mapping = {'roundnut': self.roundnut_body, 'squarenut': self.squarenut_body, 'roundpeg': self.roundpeg_body, 'squarepeg': self.squarepeg_body}

    def get_obs(self):
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body][:3])
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        goal_pick = self.env.obj_to_pick
        if goal_pick == None:
            goal_pick = 'roundnut'
        obj_to_pick_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[goal_pick]][:3])
        goal_drop = self.env.place_to_drop
        if goal_drop == None or goal_drop == 'table':
            goal_drop = 'roundpeg'
        try:
            place_to_drop_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[goal_drop]][:3])
        except KeyError:
            place_to_drop_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping['roundpeg']][:3])
        obs = np.concatenate([gripper_pos, [aperture], place_to_drop_pos, obj_to_pick_pos])
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
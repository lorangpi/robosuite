import copy
import gymnasium as gym
import robosuite as suite
import numpy as np
import os

class HanoiStateWrapper(gym.Wrapper):
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
        self.cube1_body = self.env.sim.model.body_name2id('cube1_main')
        self.cube2_body = self.env.sim.model.body_name2id('cube2_main')
        self.cube3_body = self.env.sim.model.body_name2id('cube3_main')
        self.peg1_body = self.env.sim.model.body_name2id('peg1_main')
        self.peg2_body = self.env.sim.model.body_name2id('peg2_main')
        self.peg3_body = self.env.sim.model.body_name2id('peg3_main')
        self.obj_mapping = {'cube1': self.cube1_body, 'cube2': self.cube2_body, 'cube3': self.cube3_body, 'peg1': self.peg1_body, 'peg2': self.peg2_body, 'peg3': self.peg3_body}

    def get_obs(self):
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body][:3])
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        obj_to_pick_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.env.obj_to_pick]][:3])
        goal_drop = self.env.place_to_drop
        #print("Object to pick: ", self.env.obj_to_pick, " Place to drop: ", goal_drop)
        place_to_drop_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[goal_drop]][:3])
        if 'peg' in self.env.place_to_drop:
            place_to_drop_pos = place_to_drop_pos - np.array([0.1, 0.04, 0])
        obs = np.concatenate([gripper_pos, [aperture], place_to_drop_pos, obj_to_pick_pos])
        #obs = np.concatenate([gripper_pos-obj_to_pick_pos, [aperture]])
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
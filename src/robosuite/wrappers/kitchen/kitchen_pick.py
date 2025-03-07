import copy
import gymnasium as gym
#import gym
import robosuite as suite
import numpy as np
from robosuite.utils.detector import KitchenDetector

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

class KitchenPickWrapper(gym.Wrapper):
    def __init__(self, env, render_init=False, horizon=200, image_obs=True):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.detector = KitchenDetector(self)
        self.horizon = horizon
        self.step_count = 1
        self.image_obs = image_obs
        self.obj_to_pick = None
        self.place_to_drop = None
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')

        # set up space
        self.obs_dim = self.env.obs_dim
        self.action_space = gym.spaces.Box(low=self.env.action_space.low, high=self.env.action_space.high, dtype=np.float64)

    def reset(self, seed=None):
        # Reset the environment for the drop trask
        self.obj_to_pick = np.random.choice(["pot", "bread"])
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
        success = state[f"picked_up({self.obj_to_pick})"]
        info['is_success'] = success
        truncated = truncated or self.env.done
        terminated = (terminated or success)
        self.step_count += 1
        if self.step_count > self.horizon:
            #print("Horizon reached within environment")
            terminated = True
        info["state"] = state
        reward = 10 if success else self.staged_rewards(state)
        return obs, reward, terminated, truncated, info
    
    def staged_rewards(self, state):
        """
        Calculates staged rewards based on current physical states.
        Stages consist of reaching over, doing down the button level
        """
        distances = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=True)
        obj_over = "pot_handle" if self.obj_to_pick == "pot" else self.obj_to_pick
        if state[f"over(gripper,{obj_over})"]:
            # pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id[self.obj_to_pick])][2]
            # gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][2]
            # dist = np.abs(gripper_pos - pick_pos)   
            dist = np.abs(distances[f"grasped({self.obj_to_pick})"])
            reward = -4 - (np.tanh(100.0 * dist))
        elif state[f"over(gripper,{obj_over})"] and state[f"open(gripper)"]:
            reward = -3.5
        elif state[f"over(gripper,{obj_over})"] and state[f"at_grab_level(gripper,{self.obj_to_pick})"]:
            reward = -3
        elif state[f"grasped({self.obj_to_pick})"]:
            #z_target = self.env.table_offset[2] + 0.45
            #object_z_loc = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id[self.obj_to_pick])][2]
            #z_dist = z_target - object_z_loc
            z_dist = distances[f"picked_up({self.obj_to_pick})"]
            reward = -1 - (np.tanh(10.0 * z_dist))
        else:
            #pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id[self.obj_to_pick])][:2]
            #gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:2]
            #dist = np.linalg.norm(gripper_pos - pick_pos)
            dist = distances[f"over(gripper,{obj_over})"]
            reward = -6 - (np.tanh(100.0 * dist))
        return reward
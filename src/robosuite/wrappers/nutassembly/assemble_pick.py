import copy
import gymnasium as gym
#import gym
import robosuite as suite
import numpy as np
from robosuite.utils.detector import NutAssemblyDetector

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

class AssemblePickWrapper(gym.Wrapper):
    def __init__(self, env, render_init=False, horizon=200, image_obs=True):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.detector = NutAssemblyDetector(self)
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
        self.obj_to_pick = np.random.choice(["roundnut", "squarenut"])
        self.step_count = 0
        try:
            obs, info = self.env.reset()
        except:
            obs = self.env.reset()
            info = {}    
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        info["state"] = state
        self.success_steps = 0
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
        terminated = bool(terminated)
        info["is_success"] = False
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        # Get reward
        reward = self.staged_rewards(state)
        # Check if the object has been successfully picked up
        success = state[f"picked_up({self.obj_to_pick})"]
        if success:
            self.success_steps += 1
            reward = 1000 - self.step_count*5
            if self.success_steps >= 5:  # Require 5 steps of stability
                print("Object successfully picked up", state[f"picked_up({self.obj_to_pick})"])
                info['is_success'] = True
                terminated = True
                reward = 2000 - self.step_count*10
        
        truncated = truncated or self.env.done

        self.step_count += 1
        if self.step_count > self.horizon:
            terminated = True

        info["state"] = state

        return obs, reward, terminated, truncated, info

    def staged_rewards(self, state):
        distances = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=True)
        obj_over = "pot_handle" if self.obj_to_pick == "pot" else self.obj_to_pick

        MAX_APPROACH_DIST = 0.1
        MAX_GRAB_DIST = 0.05
        MAX_PICKED_DIST = 0.02

        reward = 0  # Start with a neutral baseline

        # *** Stage 1: Success (Final Goal) ***
        if state[f"grasped({self.obj_to_pick})"]:
            z_dist = distances[f"picked_up({self.obj_to_pick})"]
            reward = 100 + 100 * (1.0 - np.clip(z_dist / MAX_PICKED_DIST, 0, 1))  # Big boost for lifting!

        # *** Stage 2: Gripper at Correct Grab Level ***
        elif state[f"over(gripper,{obj_over})"] and state[f"at_grab_level(gripper,{self.obj_to_pick})"]:
            grab_level_dist = distances[f"at_grab_level(gripper,{self.obj_to_pick})"]
            reward = 10 + 30 * (1.0 - np.clip(grab_level_dist / MAX_GRAB_DIST, 0, 1))  # Reward being at grab level

            if state[f"open_gripper(gripper)"]:
                reward += 20  # Encourage keeping gripper open before grasping

        # *** Stage 3: Getting Near the Object (Approaching) ***
        else:
            approach_dist = distances[f"over(gripper,{obj_over})"]
            reward = 2 * (1.0 - np.clip(approach_dist / MAX_APPROACH_DIST, 0, 1))  # Reward approaching smoothly

        return reward

    # def staged_rewards(self, state):
    #     """
    #     Calculates staged rewards based on current physical states.
    #     Stages consist of reaching over, doing down the button level
    #     """
    #     MAX_APPROACH_DIST = 0.5   # maximum distance for approaching the object
    #     MAX_GRAB_DIST = 0.2       # maximum distance considered for grab-level alignment
    #     MAX_PICKED_DIST = 0.1     # maximum distance for the picked-up stage
    #     distances = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=True)
    #     obj_over = "pot_handle" if self.obj_to_pick == "pot" else self.obj_to_pick
    #     if state[f"grasped({self.obj_to_pick})"]:
    #         z_dist = distances[f"picked_up({self.obj_to_pick})"]
    #         reward = 4 - np.clip(dist/MAX_PICKED_DIST, 0, 1)#np.tanh(20.0 * z_dist)
    #     elif state[f"over(gripper,{obj_over})"] and state[f"at_grab_level(gripper,{self.obj_to_pick})"] and state[f"open_gripper(gripper)"]:
    #         reward = 3
    #     elif state[f"over(gripper,{obj_over})"] and state[f"open_gripper(gripper)"]:
    #         dist = distances[f"at_grab_level(gripper,{self.obj_to_pick})"]
    #         reward = 2 - np.clip(dist/MAX_GRAB_DIST, 0, 1)#np.tanh(20.0 * distances[f"at_grab_level(gripper,{self.obj_to_pick})"])
    #     elif state[f"over(gripper,{obj_over})"]:
    #         aperture = distances[f"open_gripper(gripper)"]
    #         reward = aperture
    #     else:
    #         #pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id[self.obj_to_pick])][:2]
    #         #gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:2]
    #         #dist = np.linalg.norm(gripper_pos - pick_pos)
    #         dist = distances[f"over(gripper,{obj_over})"]
    #         reward = - np.clip(dist/MAX_APPROACH_DIST, 0, 1)#(np.tanh(10.0 * dist))
    #     return reward
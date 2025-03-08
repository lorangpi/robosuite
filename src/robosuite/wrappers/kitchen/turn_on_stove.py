import copy
import gymnasium as gym
#import gym
import robosuite as suite
import numpy as np
from robosuite.utils.detector import KitchenDetector


class TurnOnStoveWrapper(gym.Wrapper):
    def __init__(self, env, render_init=False, horizon=100):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.detector = KitchenDetector(self)
        self.step_count = 0
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.horizon = horizon
        self.obj_to_pick = None
        self.place_to_drop = None

        # set up spaces
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(low=self.env.action_space.low, high=self.env.action_space.high, dtype=np.float64)
        #print("Action space: ", self.action_space)

    def reset(self, seed=None):
        # Reset the environment for the drop trask
        try:
            obs, info = self.env.reset()
        except:
            obs = self.env.reset()
            info = {}    
        # Run 4 random actions to randomize the initial state
        for _ in range(8):
            action = self.action_space.sample()
            obs, _, _, _, _ = self.env.step(action)
        self.step_count = 0
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
        success = state[f"stove_on()"]
        info['is_success'] = success
        truncated = truncated or self.env.done
        terminated = (terminated or success)
        self.step_count += 1
        if self.step_count > self.horizon:
            #print("Horizon reached within environment")
            terminated = True
        info["state"] = state
        reward = 1000 if success else self.staged_rewards(state)
        return obs, reward, terminated, truncated, info
    
    def staged_rewards(self, state):
        """
        Calculates staged rewards based on current physical states.
        Stages consist of reaching over, doing down the button level
        """
        if state[f"over(gripper,button)"] and state[f"at_grab_level(gripper,button)"]:
            reward = -0.5
        elif state[f"over(gripper,button)"]:
            pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id["button"])][2]
            gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][2]
            dist = np.abs(gripper_pos - pick_pos)   
            reward = -2 - (np.tanh(20.0 * dist))
        else:
            pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id["button"])][:2]
            gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:2]
            dist = np.linalg.norm(gripper_pos - pick_pos)
            reward = -3 - (np.tanh(10.0 * dist))
        return reward
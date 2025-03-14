import copy
import gymnasium as gym
#import gym
import robosuite as suite
import numpy as np
from robosuite.utils.detector import KitchenDetector


class TurnOnStoveWrapper(gym.Wrapper):
    def __init__(self, env, render_init=False, horizon=100, image_obs=True):
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
        success = state[f"stove_on()"]
        if success:
            self.success_steps += 1
            if self.success_steps >= 5:  # Require 5 steps of stability
                print("Stove successfully turned off")
                info['is_success'] = True
                terminated = True
                reward += 1000 - self.step_count*5
        
        truncated = truncated or self.env.done

        self.step_count += 1
        if self.step_count > self.horizon:
            terminated = True
        
        info["state"] = state

        return obs, reward, terminated, truncated, info
    
    def staged_rewards(self, state):
        """
        Reward shaping for turning off a stove button in stages:
        Stage 1: Approach the button (XY distance)
        Stage 2: Vertical alignment (getting close to the button's press height)
        Stage 3: Precise press (final stage; can serve as terminal if button_off flag exists)
        """
        # Define maximum distances for scaling rewards
        MAX_APPROACH_DIST = 0.5   # max allowed XY distance to button for approaching
        MAX_PRESS_DIST = 0.1      # vertical distance threshold to consider as pressed

        reward = 0  # baseline

        # --- (Optional) Terminal check: if a flag indicates the button is off ---
        # if state.get("button_off", False):
        #     return 100  # large reward for achieving the task

        # Stage 1: Approaching the button (XY plane)
        if not state[f"over(gripper,button)"]:
            # Get XY positions for gripper and button
            button_xy = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id["button"])][:2]
            gripper_xy = self.env.sim.data.body_xpos[self.gripper_body][:2]
            dist_xy = np.linalg.norm(gripper_xy - button_xy)
            # Negative reward (penalty) that lessens as the gripper gets closer
            reward = -10 * np.clip(dist_xy / MAX_APPROACH_DIST, 0, 1)
        
        # If the gripper is over the button (good XY alignment)
        else:
            # Stage 2: Check vertical alignment (Z-axis)
            button_z = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id["button"])][2]
            gripper_z = self.env.sim.data.body_xpos[self.gripper_body][2]
            vertical_dist = np.abs(gripper_z - button_z)

            # If also at the proper press level (you can use the same flag as in your original function)
            if state[f"at_grab_level(gripper,button)"]:
                # Stage 3: Precise press stage
                # Reward increases as vertical distance is minimized.
                reward = 50 + 50 * (1.0 - np.clip(vertical_dist / MAX_PRESS_DIST, 0, 1))
            else:
                # If over the button but not yet at press level, encourage reducing vertical gap.
                reward = 10 * (1.0 - np.clip(vertical_dist / MAX_PRESS_DIST, 0, 1))
        
        return reward

    # def staged_rewards(self, state):
    #     """
    #     Calculates staged rewards based on current physical states.
    #     Stages consist of reaching over, doing down the button level
    #     """
    #     MAX_APPROACH_DIST = 0.5   # maximum distance for approaching the object
    #     MAX_GRAB_DIST = 0.2       # maximum distance considered for grab-level alignment
    #     if state[f"over(gripper,button)"] and state[f"at_grab_level(gripper,button)"]:
    #         reward = 2
    #     elif state[f"over(gripper,button)"]:
    #         pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id["button"])][2]
    #         gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][2]
    #         dist = np.abs(gripper_pos - pick_pos)   
    #         reward = 1 - np.clip(dist/MAX_GRAB_DIST, 0, 1)#(np.tanh(20.0 * dist))
    #     else:
    #         pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id["button"])][:2]
    #         gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:2]
    #         dist = np.linalg.norm(gripper_pos - pick_pos)
    #         reward = - np.clip(dist/MAX_APPROACH_DIST, 0, 1)#(np.tanh(10.0 * dist))
    #     return reward
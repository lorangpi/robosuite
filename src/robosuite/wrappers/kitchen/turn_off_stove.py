import copy
import gymnasium as gym
#import gym
import robosuite as suite
import numpy as np
from robosuite.utils.detector import KitchenDetector


class TurnOffStoveWrapper(gym.Wrapper):
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

    def cap(self, eps, max_val=0.12, min_val=0.01):
        """
        Caps the displacement
        """
        # If the displacement is greater than the max value, cap it
        if np.linalg.norm(eps) > max_val:
            eps = eps / np.linalg.norm(eps) * max_val
        # If the displacement is smaller than the min value, cap it
        if np.linalg.norm(eps) < min_val:
            eps = eps / np.linalg.norm(eps) * min_val
        return eps

    def reset_button_on(self, state):
        """
        Transitons the environment to a state where button is turned on
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        #print("\n\n\t\t---------------------PICK GOAL IS: ", goal)
        #print("\n\n")

        goal_str = "button"
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        goal_pos = self.env.sim.data.body_xpos[goal][:3]
        goal_quat = self.env.sim.data.body_xquat[goal]
        self.keypoint = np.concatenate([goal_pos, goal_quat])

        reset_step_count = 0

        #print("Moving gripper over button...")
        while not state['over(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]])
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs, info
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to swith level ...")
        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]])
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs, info
        reset_step_count = 0
        self.env.time_step = 0

        #print("Turning on button...")
        while not state['stove_on()']:
            action = np.asarray([0,0.3,0,0])
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 50:
                return False, obs, info
        reset_step_count = 0
        self.env.time_step = 0

        for _ in range(15):
            action = np.array([0, 0, 1, 0])
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            obs, state = next_obs, next_state
            reset_step_count += 1

        return True, obs, info

    def reset(self, seed=None):
        # Reset the environment for the drop trask
        self.step_count = 0
        success_reset = False
        while not success_reset:
            try:
                obs, info = self.env.reset()
            except:
                obs = self.env.reset()
                info = {}    
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            success_reset, obs, info = self.reset_button_on(state)
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
        success = not(state[f"stove_on()"])
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
        if state[f"over(gripper,button)"]:
            pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id["button"])][2]
            gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][2]
            dist = np.abs(gripper_pos - pick_pos)   
            reward = -2 - (np.tanh(10.0 * dist))
        elif state[f"over(gripper,button)"] and state[f"at_grab_level(gripper,button)"]:
            reward = -0.5
        else:
            pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id["button"])][:2]
            gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:2]
            dist = np.linalg.norm(gripper_pos - pick_pos)
            reward = -3 - (np.tanh(10.0 * dist))
        return reward
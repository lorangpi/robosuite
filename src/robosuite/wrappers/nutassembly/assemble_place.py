import copy
import gymnasium as gym
#import gym
import robosuite as suite
import numpy as np
from robosuite.utils.detector import NutAssemblyDetector

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

class AssemblePlaceWrapper(gym.Wrapper):
    def __init__(self, env, render_init=False, horizon=200, image_obs=True):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.render_init = render_init
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

    def reset_pick(self, state):
        """
        Transitons the environment to a state where the gripper is has picked the object
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        #print("\n\n\t\t---------------------PICK GOAL IS: ", goal)
        #print("\n\n")
        reset_step_count = 0

        goal_str = self.obj_to_pick
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        # Moving gripper 10 cm above the object
        ref_z = 1.1
        z_pos = 0
        while z_pos < ref_z:
            z_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])[2]
            dist_z = abs(ref_z - z_pos)
            dist_z = self.cap([dist_z])
            action = 5*np.concatenate([[0, 0], dist_z, [0]])
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs, info
            dist_z = dist_z[0]
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving gripper over object...")
        gripper_goal = "pot_handle" if goal_str == "pot" else goal_str
    
        while not state['over(gripper,{})'.format(gripper_goal)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0, 0.05, 0])
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

        #print("Opening gripper...")
        while not state['open_gripper(gripper)']:
            action = np.asarray([0,0,0,-1])
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 100:
                return False, obs, info
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to grab level...")
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

        #print("Closing gripper...")
        #while not state['grasped({})'.format(goal_str)]:
        for _ in range(10):
            action = np.asarray([0,0,0,1])
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            reset_step_count += 1
            if reset_step_count > 50:
                return False, obs, info
        reset_step_count = 0
        self.env.time_step = 0

        #print("Lifting object...")
        while not state['picked_up({})'.format(goal_str)]:
            action = np.asarray([0,0,0.4,0])
            action = 5*self.cap(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 300:
                return False, obs, info
        reset_step_count = 0
        self.env.time_step = 0

        return True, obs, info


    def reset(self, seed=None):
        # Reset the environment for the drop trask
        self.obj_to_pick = np.random.choice(["roundnut", "squarenut"])
        if self.obj_to_pick == "roundnut":
            self.place_to_drop = "roundpeg"
        else:
            self.place_to_drop = "squarepeg"
        self.step_count = 0
        success_reset = False
        while not success_reset:
            try:
                obs, info = self.env.reset()
            except:
                obs = self.env.reset()
                info = {}    
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            success_reset, obs, info = self.reset_pick(state)
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
        success = state[f"on({self.obj_to_pick},{self.place_to_drop})"] and not(state[f"grasped({self.obj_to_pick})"])
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
        if state[f"on({self.obj_to_pick},{self.place_to_drop})"]:
            reward = -2
        elif state[f"over(gripper,{self.place_to_drop})"]:
            drop_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id[self.place_to_drop])][2]
            gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][2]
            dist = np.linalg.norm(drop_pos - gripper_pos)
            reward = -4 - (np.tanh(100.0 * dist))
        elif state[f"picked_up({self.obj_to_pick})"]:
            drop_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id[self.place_to_drop])][:2]
            gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:2]
            dist = np.linalg.norm(drop_pos - gripper_pos)
            reward = -6 - (np.tanh(100.0 * dist))
        elif not(state[f"grasped({self.obj_to_pick})"]):
            pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id[self.obj_to_pick])][:3]
            gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:3]
            dist = np.linalg.norm(pick_pos - gripper_pos)
            reward = -20 - (np.tanh(100.0 * dist))
        else:
            pick_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id[self.obj_to_pick])][2]
            table_pos = self.env.table_offset[2] + 0.45
            dist = np.linalg.norm(pick_pos - table_pos)
            reward = -8 - (np.tanh(100.0 * dist))
        return reward
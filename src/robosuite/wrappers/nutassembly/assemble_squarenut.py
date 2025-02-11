import copy
import gymnasium as gym
#import gym
import robosuite as suite
import numpy as np
from robosuite.utils.detector import NutAssemblyDetector


class AssembleSquareNutWrapper(gym.Wrapper):
    def __init__(self, env):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.detector = NutAssemblyDetector(self)
        self.step_count = 0
        self.obj_to_pick = "squarenut"
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')

        # set up spaces
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(low=self.env.action_space.low, high=self.env.action_space.high, dtype=np.float64)
        #print("Action space: ", self.action_space)

    def reset_grasp(self, state):
        counter = 0
        while not state['over(gripper,{})'.format(self.obj_to_pick)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(f'{self.obj_to_pick}_main')])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            action = 5*np.concatenate([dist_xy_plan, [0, 0]])
            next_obs, reward, _, _, info = self.env.step(action)
            self.env.render()
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            counter += 1
            if counter > 100:
                return False, next_obs, info

        counter = 0
        # Shift slightely to the right
        #print("Shifting slightly to the left...")
        for _ in range(10):
            action = np.asarray([0,0.5,0,0])
            action = action
            next_obs, reward, _, _, info = self.env.step(action)
            self.env.render()
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

        #print("Opening gripper...")
        while not state['open_gripper(gripper)']:
            action = np.asarray([0,0,0,-1])
            action = action
            next_obs, reward, _, _, info = self.env.step(action)
            self.env.render()
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            counter += 1
            if counter > 100:
                return False, next_obs, info
        counter = 0

        #print("Moving down gripper to grab level...")
        while not state['at_grab_level(gripper,{})'.format(self.obj_to_pick)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(f'{self.obj_to_pick}_main')])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]])
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render()
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            counter += 1
            if counter > 100:
                return False, next_obs, info
        counter = 0

        #print("Closing gripper...")
        while not state['grasped({})'.format(self.obj_to_pick)]:
            action = np.asarray([0,0,0,1])
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render()
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            counter += 1
            if counter > 100:
                return False, next_obs, info
        #print("Grasped object")
        return True, next_obs, info

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
            success_reset, obs, info = self.reset_grasp(state)
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
        success = state[f"on(squarenut,squarepeg)"]
        info['is_success'] = success
        truncated = truncated or self.env.done
        terminated = (terminated or success)
        self.step_count += 1
        if self.step_count > self.horizon:
            #print("Horizon reached within environment")
            terminated = True
        info["state"] = state
        reward = max(self.staged_rewards(state))
        reward += 10 if success else 0
        #if success or self.step_count%10 == 0:
            #print(reward)
        return obs, reward, terminated, truncated, info
    
    def staged_rewards(self, state):
        """
        Calculates staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already on the correct pegs
        active_nuts = [self.env.nuts[0]] # corresponds to the square nut

        # reaching reward governed by distance to closest object
        r_reach = 0.0
        if active_nuts:
            # reaching reward via minimum distance to the handles of the objects
            dists = np.array(
                [
                    np.linalg.norm(
                        self.sim.data.body_xpos[self.obj_body_id[active_nut.name]]
                        - self.sim.data.body_xpos[self.peg2_body_id]
                    )
                    for active_nut in active_nuts
                ]
            )
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        # grasping reward for touching any objects of interest
        is_grasped = state["grasped(squarenut)"]
        r_grasp = (
            int(
                is_grasped
            )
            * grasp_mult
        )

        # lifting reward for picking up an object
        r_lift = 0.0
        table_pos = np.array(self.sim.data.body_xpos[self.table_body_id])
        if active_nuts and r_grasp > 0.0:
            z_target = table_pos[2] + 0.2
            object_z_locs = self.sim.data.body_xpos[[self.obj_body_id[active_nut.name] for active_nut in active_nuts]][
                :, 2
            ]
            z_dists = np.maximum(z_target - object_z_locs, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (lift_mult - grasp_mult)

        # hover reward for getting object above peg
        r_hover = 0.0
        if active_nuts:
            r_hovers = np.zeros(len(active_nuts))
            peg_body_ids = [self.peg1_body_id, self.peg2_body_id]
            for i, nut in enumerate(active_nuts):
                valid_obj = False
                peg_pos = None
                for nut_name, idn in self.nut_to_id.items():
                    if nut_name in nut.name.lower():
                        peg_pos = np.array(self.sim.data.body_xpos[peg_body_ids[idn]])[:2]
                        valid_obj = True
                        break
                if not valid_obj:
                    raise Exception("Got invalid object to reach: {}".format(nut.name))
                ob_xy = self.sim.data.body_xpos[self.obj_body_id[nut.name]][:2]
                dist = np.linalg.norm(peg_pos - ob_xy)
                r_hovers[i] = r_lift + (1 - np.tanh(10.0 * dist)) * (hover_mult - lift_mult)
            r_hover = np.max(r_hovers)

        return r_reach, r_grasp, r_lift, r_hover
import gymnasium as gym
import robosuite as suite
import numpy as np
from stable_baselines3 import SAC
from detector import Robosuite_Hanoi_Detector

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

class DropWrapper(gym.Wrapper):
    def __init__(self, env, render_init=False):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.render_init = render_init
        self.detector = Robosuite_Hanoi_Detector(self)

        # Define needed variables
        self.cube1_body = self.env.sim.model.body_name2id('cube1_main')
        self.cube2_body = self.env.sim.model.body_name2id('cube2_main')
        self.cube3_body = self.env.sim.model.body_name2id('cube3_main')
        self.peg1_body = self.env.sim.model.body_name2id('peg1_main')
        self.peg2_body = self.env.sim.model.body_name2id('peg2_main')
        self.peg3_body = self.env.sim.model.body_name2id('peg3_main')

        # Set reset state info:
        self.reset_state = {'on(cube1,peg1)': True, 'on(cube2,peg3)': True, 'on(cube3,peg2)': True}
        self.env.reset_state = self.reset_state
        self.obj_to_pick = "cube1"
        self.place_to_drop = "cube3"
        self.obj_mapping = {'cube1': self.cube1_body, 'cube2': self.cube2_body, 'cube3': self.cube3_body, 'peg1': self.peg1_body, 'peg2': self.peg2_body, 'peg3': self.peg3_body}

        # set up observation space
        self.obs_dim = self.env.obs_dim #+ 6 # 6 extra dimensions for the distance to objects/areas

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    def drop_reset(self):
        """
        Resets the environment to a state where the gripper is holding the object on top of the drop-off location
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state

        self.reset_step_count = 0
        #print("Moving up...")
        for _ in range(5):
            obs,_,_,_,_ = self.env.step([0,0,1,0])
            self.env.render() if self.render_init else None

        #print("Moving gripper over object...")
        while not state['over(gripper,{})'.format(self.obj_to_pick)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            action = 5*np.concatenate([dist_xy_plan, [0, 0]])
            obs,_,_,_,_ = self.env.step(action)
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 500:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Opening gripper...")
        while not state['open_gripper(gripper)']:
            obs,_,_,_,_ = self.env.step([0,0,0,-0.1])
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 100:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to grab level...")
        while not state['at_grab_level(gripper,{})'.format(self.obj_to_pick)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]])
            obs,_,_,_,_ = self.env.step(action)
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 400:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Closing gripper...")
        while not state['grasped({})'.format(self.obj_to_pick)]:
            obs,_,_,_,_ = self.env.step([0,0,0,0.1])
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 30:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Lifting object...")
        while not state['picked_up({})'.format(self.obj_to_pick)]:
            obs,_,_,_,_ = self.env.step([0,0,0.4,0])
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 300:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving gripper over place to drop...")
        while not state['over(gripper,{})'.format(self.place_to_drop)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            if 'peg' in self.place_to_drop:
                object_pos = self.area_pos[self.place_to_drop]
            else:
                object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            action = 5*np.concatenate([dist_xy_plan, [0, 0]])
            obs,_,_,_,_ = self.env.step(action)
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 500:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        return True, obs

    def reset(self, seed=None):
        # Reset the environment for the drop trask
        self.env.set_reset_state(self.reset_state)
        success = False
        while not success:
            try:
                obs, info = self.env.reset(seed=seed)
            except:
                obs = self.env.reset(seed=seed)
                info = {}
            success, obs = self.drop_reset()

        self.sim.forward()
        return obs, info
    
    def step(self, action):
        truncated = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        state = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)
        success = state[f"on({self.obj_to_pick},{self.place_to_drop})"]
        info['is_sucess'] = success
        truncated = truncated or self.env.done
        terminated = terminated or success
        return obs, reward, terminated, truncated, info

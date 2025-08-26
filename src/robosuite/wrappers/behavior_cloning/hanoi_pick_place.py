import copy
#import gymnasium as gym
import gym
import robosuite as suite
import numpy as np
from robosuite.wrappers.behavior_cloning.detector import Robosuite_Hanoi_Detector

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

class PickPlaceWrapper(gym.Wrapper):
    def __init__(self, env, render_init=False, nulified_action_indexes=[], horizon=2000, hanoi=False, oracle=True):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.render_init = render_init
        self.detector = Robosuite_Hanoi_Detector(self)
        self.nulified_action_indexes = nulified_action_indexes
        self.horizon = horizon
        self.step_count = 1
        self.hanoi = hanoi 
        self.oracle = oracle

        # Define needed variables
        self.cube1_body = self.env.sim.model.body_name2id('cube1_main')
        self.cube2_body = self.env.sim.model.body_name2id('cube2_main')
        self.cube3_body = self.env.sim.model.body_name2id('cube3_main')
        self.peg1_body = self.env.sim.model.body_name2id('peg1_main')
        self.peg2_body = self.env.sim.model.body_name2id('peg2_main')
        self.peg3_body = self.env.sim.model.body_name2id('peg3_main')
        # Set reset state info:
        #self.reset_state = {'on(cube1,peg1)': True, 'on(cube2,peg3)': True, 'on(cube3,peg2)': True}
        self.reset_state = self.sample_reset_state()
        self.task = self.sample_task()
        self.env.reset_state = self.reset_state
        self.obj_mapping = {'cube1': self.cube1_body, 'cube2': self.cube2_body, 'cube3': self.cube3_body, 'peg1': self.peg1_body, 'peg2': self.peg2_body, 'peg3': self.peg3_body}
        self.goal_mapping = {'cube1': 0, 'cube2': 1, 'cube3': 2, 'peg1': 3, 'peg2': 4, 'peg3': 5}
        self.area_pos = {'peg1': self.env.pegs_xy_center[0], 'peg2': self.env.pegs_xy_center[1], 'peg3': self.env.pegs_xy_center[2]}
        if env.env_id == 'Hanoi4x3':
            self.detector.objects = ['cube1', 'cube2', 'cube3', 'cube4']
            self.detector.object_id = {'cube1': 'cube1_main', 'cube2': 'cube2_main', 'cube3': 'cube3_main', 'cube4': 'cube4_main', 'peg1': 'peg1_main', 'peg2': 'peg2_main', 'peg3': 'peg3_main'}
            self.detector.object_areas = ['peg1', 'peg2', 'peg3']
            self.cube4_body = self.env.sim.model.body_name2id('cube4_main')
            self.obj_mapping['cube4'] = self.cube4_body
        if env.env_id == 'Hanoi4x4':
            self.detector.objects = ['cube1', 'cube2', 'cube3', 'cube4']
            self.detector.object_id = {'cube1': 'cube1_main', 'cube2': 'cube2_main', 'cube3': 'cube3_main', 'cube4': 'cube4_main', 'peg0': 'peg0_main', 'peg1': 'peg1_main', 'peg2': 'peg2_main', 'peg3': 'peg3_main', 'peg4': 'peg4_main'}
            self.detector.object_areas = ['peg0', 'peg1', 'peg2', 'peg3']
            self.peg0_body = self.env.sim.model.body_name2id('peg0_main')
            self.obj_mapping['peg0'] = self.peg0_body
            self.detector.area_pos = {'peg0': self.env.pegs_xy_center[0], 'peg1': self.env.pegs_xy_center[1], 'peg2': self.env.pegs_xy_center[2], 'peg3': self.env.pegs_xy_center[3]}
            self.cube4_body = self.env.sim.model.body_name2id('cube4_main')
            self.obj_mapping['cube4'] = self.cube4_body
        if env.env_id == 'Hanoi5x4':
            self.detector.objects = ['cube1', 'cube2', 'cube3', 'cube4', 'cube5']
            self.detector.object_id = {'cube1': 'cube1_main', 'cube2': 'cube2_main', 'cube3': 'cube3_main', 'cube4': 'cube4_main', 'cube5': 'cube5_main', 'peg0': 'peg0_main', 'peg1': 'peg1_main', 'peg2': 'peg2_main', 'peg3': 'peg3_main', 'peg4': 'peg4_main'}
            self.detector.object_areas = ['peg0', 'peg1', 'peg2', 'peg3']
            self.peg0_body = self.env.sim.model.body_name2id('peg0_main')
            self.obj_mapping['peg0'] = self.peg0_body
            self.detector.area_pos = {'peg0': self.env.pegs_xy_center[0], 'peg1': self.env.pegs_xy_center[1], 'peg2': self.env.pegs_xy_center[2], 'peg3': self.env.pegs_xy_center[3]}
            self.cube4_body = self.env.sim.model.body_name2id('cube4_main')
            self.obj_mapping['cube4'] = self.cube4_body
            self.cube5_body = self.env.sim.model.body_name2id('cube5_main')
            self.obj_mapping['cube5'] = self.cube5_body
        if env.env_id == 'Hanoi5x5':
            self.detector.objects = ['cube1', 'cube2', 'cube3', 'cube4', 'cube5']
            self.detector.object_id = {'cube1': 'cube1_main', 'cube2': 'cube2_main', 'cube3': 'cube3_main', 'cube4': 'cube4_main', 'cube5': 'cube5_main', 'peg0': 'peg0_main', 'peg1': 'peg1_main', 'peg2': 'peg2_main', 'peg3': 'peg3_main', 'peg4': 'peg4_main'}
            self.detector.object_areas = ['peg0', 'peg1', 'peg2', 'peg3', 'peg4']
            self.peg0_body = self.env.sim.model.body_name2id('peg0_main')
            self.obj_mapping['peg0'] = self.peg0_body
            self.peg4_body = self.env.sim.model.body_name2id('peg4_main')
            self.obj_mapping['peg4'] = self.peg4_body
            self.detector.area_pos = {'peg0': self.env.pegs_xy_center[0], 'peg1': self.env.pegs_xy_center[1], 'peg2': self.env.pegs_xy_center[2], 'peg3': self.env.pegs_xy_center[3], 'peg4': self.env.pegs_xy_center[4]}
            self.cube4_body = self.env.sim.model.body_name2id('cube4_main')
            self.obj_mapping['cube4'] = self.cube4_body
            self.cube5_body = self.env.sim.model.body_name2id('cube5_main')
            self.obj_mapping['cube5'] = self.cube5_body
        if env.env_id == 'Hanoi7x5':
            self.detector.objects = ['cube1', 'cube2', 'cube3', 'cube4', 'cube5', 'cube6', 'cube7']
            self.detector.object_id = {'cube1': 'cube1_main', 'cube2': 'cube2_main', 'cube3': 'cube3_main', 'cube4': 'cube4_main', 'cube5': 'cube5_main', 'cube6': 'cube6_main', 'cube7': 'cube7_main', 'peg0': 'peg0_main', 'peg1': 'peg1_main', 'peg2': 'peg2_main', 'peg3': 'peg3_main', 'peg4': 'peg4_main'}
            self.detector.object_areas = ['peg0', 'peg1', 'peg2', 'peg3', 'peg4']
            self.peg0_body = self.env.sim.model.body_name2id('peg0_main')
            self.obj_mapping['peg0'] = self.peg0_body
            self.peg4_body = self.env.sim.model.body_name2id('peg4_main')
            self.obj_mapping['peg4'] = self.peg4_body
            self.detector.area_pos = {'peg0': self.env.pegs_xy_center[0], 'peg1': self.env.pegs_xy_center[1], 'peg2': self.env.pegs_xy_center[2], 'peg3': self.env.pegs_xy_center[3], 'peg4': self.env.pegs_xy_center[4]}
            self.cube4_body = self.env.sim.model.body_name2id('cube4_main')
            self.obj_mapping['cube4'] = self.cube4_body
            self.cube5_body = self.env.sim.model.body_name2id('cube5_main')
            self.obj_mapping['cube5'] = self.cube5_body
            self.cube6_body = self.env.sim.model.body_name2id('cube6_main')
            self.obj_mapping['cube6'] = self.cube6_body
            self.cube7_body = self.env.sim.model.body_name2id('cube7_main')
            self.obj_mapping['cube7'] = self.cube7_body
            
        # set up observation space
        self.obs_dim = 10

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)
        if self.nulified_action_indexes != []:
            self.action_space = gym.spaces.Box(low=self.env.action_space.low[:-len(nulified_action_indexes)], high=self.env.action_space.high[:-len(nulified_action_indexes)], dtype=np.float64)
        else:
            self.action_space = gym.spaces.Box(low=self.env.action_space.low, high=self.env.action_space.high, dtype=np.float64)

    def search_free_space(self, cube, locations, reset_state):
        drop_off = np.random.choice(locations)
        reset_state.update({f"on({cube},{drop_off})":True})
        locations.remove(drop_off)
        locations.append(cube)
        return reset_state, locations

    def sample_reset_state(self):
        reset_state = {}
        locations = ["peg1", "peg2", "peg3"]
        cubes = ["cube3", "cube2", "cube1"]
        for cube in cubes:
            reset_state, locations = self.search_free_space(cube, locations=locations, reset_state=reset_state)
        return reset_state

    def search_valid_picks_drops(self):
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        valid_picks = []
        cubes = [3, 2, 1]
        pegs = [4, 5, 6]
        for cube in cubes:
            if state[f"clear(cube{cube})"]:
                valid_picks.append(cube)
        valid_drops = copy.copy(valid_picks)
        for peg in pegs:
            if state[f"clear(peg{peg-3})"]:
                valid_drops.append(peg)
        return valid_picks, valid_drops
    
    def sample_task(self):
        # Sample a random task
        valid_task = False
        valid_picks, valid_drops = self.search_valid_picks_drops()

        while not valid_task:
            # Sample a random task
            cube_to_pick = np.random.choice(valid_picks)
            valid_drops_copy = copy.copy(valid_drops)
            valid_drops_copy.remove(cube_to_pick)
            place_to_drop = np.random.choice(valid_drops_copy)
            if cube_to_pick >= place_to_drop:
                continue
            if place_to_drop < 4:
                place_to_drop = 'cube{}'.format(place_to_drop)
            else:
                place_to_drop = 'peg{}'.format(place_to_drop - 3)
            cube_to_pick = 'cube{}'.format(cube_to_pick)
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            if state['on({},{})'.format(cube_to_pick, place_to_drop)]:
                continue
            if state['clear({})'.format(cube_to_pick)] and state['clear({})'.format(place_to_drop)]:
                valid_task = True
        # Set the task
        self.obj_to_pick = cube_to_pick
        self.place_to_drop = place_to_drop
        print("Task: Pick {} and drop it on {}".format(self.obj_to_pick, self.place_to_drop))
        #return f"on({cube_to_pick},{place_to_drop})"
        return (cube_to_pick, place_to_drop)

    def reach_pick_reset(self):
        """
        Resets the environment to a state where the gripper is holding the object on top of the drop-off location
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state

        self.reset_step_count = 0
        # Moving randomly 0 to 50 steps
        for k in range(np.random.randint(1, 50)):
            generate_random_3d_action = np.random.uniform(-0.2, 0.2, 3)
            action = np.concatenate([generate_random_3d_action, [0]])
            obs,_,_,_,_ = self.env.step(action)
            self.env.render() if self.render_init else None

        #print("Moving gripper over object...")
        # while not state['over(gripper,{})'.format(self.place_to_drop)]:
        #     gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
        #     object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]])
        #     dist_xy_plan = object_pos[:2] - gripper_pos[:2]
        #     action = 5*np.concatenate([dist_xy_plan, [0, 0]])
        #     obs,_,_,_,_ = self.env.step(action)
        #     self.env.render() if self.render_init else None
        #     state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        #     self.reset_step_count += 1
        #     if self.reset_step_count > 500:
        #         return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        return True, obs

    def valid_state(self):
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        state = {k: state[k] for k in state if 'on' in k}
        # Filter only the values that are True
        state = {key: value for key, value in state.items() if value}
        # if state has not 3 keys, return None
        #print(state)
        #print(len(state))
        if "4x" in self.env.env_id:
            if len(state) != 4:
                return False
        elif "5x" in self.env.env_id:
            if len(state) != 5:
                return False
        elif "7x" in self.env.env_id:
            if len(state) != 7:
                return False
        else:
            if len(state) != 3:
                return False
        # Check if cubes have fallen from other subes, i.e., check if two or more cubes are on the same peg
        pegs = []
        for relation, value in state.items():
            _, peg = relation.split('(')[1].split(',')
            pegs.append(peg)
        if len(pegs) != len(set(pegs)):
            #print("Two or more cubes are on the same peg")
            return False
        #print(state)
        return True

    def reset(self, seed=None):
        # Reset the environment for the drop trask
        self.step_count = 1
        reset = False
        while not reset:
            trials = 0
            self.reset_state = self.sample_reset_state()
            self.env.reset_state = self.reset_state
            success = False
            while not success:
                valid_state = False
                while not valid_state:
                    #print("Trying to reset the environment...")
                    try:
                        obs, info = self.env.reset()
                    except:
                        obs = self.env.reset()
                        info = {}
                    valid_state = self.valid_state()
                    trials += 1
                    if trials > 3:
                        break   
                # 3 times out of 4, the env is reset to another location
                if np.random.rand() < 0.25:
                    success = True
                else:
                    success, obs = self.reach_pick_reset()
                reset = success
                if trials > 3:
                    break   
            self.task = self.sample_task()

        self.sim.forward()
        self.goal = self.obj_to_pick
        #obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3]))
        if self.oracle:
            obs = self.filter_obs(obs)
        else:
            obs = self.simple_obs(obs)
        goal_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3]
        #goal_quat = self.env.sim.data.body_xquat[self.obj_mapping[self.goal]]

        #self.keypoint = np.concatenate([goal_pos, goal_quat])
        self.keypoint = goal_pos
        info["keypoint"] = self.keypoint
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        info["state"] = state
        info["task"] = self.task
        #print("Task: Pick {} and drop it on {}".format(self.obj_to_pick, self.place_to_drop))
        return obs, info

    # def filter_obs(self, obs):
    #     # Filter the observations to only include the relevant information
    #     # If cube1 is the object to pick, then the observation should only include the position and quat of cube1
    #     # cube1: obs[0:7], cube2: obs[7:14], cube3: obs[14:21] and rest of the obs (21::)
    #     map_cube_obs = {"cube1": obs[0:7], "cube2": obs[7:14], "cube3": obs[14:21]}
    #     gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body][:3])
    #     gripper_quat = np.asarray(self.env.sim.data.body_xquat[self.env.gripper_body])
    #     left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
    #     right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
    #     aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
    #     #aperture = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)['open_gripper(gripper)']
    #     if 'cube' in self.goal:
    #         #return np.concatenate([map_cube_obs[self.goal], obs[21:]])
    #         #return np.concatenate([map_cube_obs[self.goal], gripper_pos, gripper_quat, [aperture]])
    #         return np.concatenate([map_cube_obs[self.goal][:3], gripper_pos + np.array([0.05, 0, 0]), [aperture]])
    #     elif 'peg' in self.goal:
    #         peg_pos = self.env.env.sim.data.body_xpos[self.obj_mapping[self.goal]][:3] - np.array([0.1, 0.04, 0])
    #         peg_pos = np.concatenate([peg_pos, [0, 0, 0, 1]])
    #         #return np.concatenate([peg_pos, obs[21:]])
    #         #return np.concatenate([peg_pos, gripper_pos, gripper_quat, [aperture]])
    #         return np.concatenate([peg_pos[:3], gripper_pos, [aperture]])

    def set_task(self, task):
        self.obj_to_pick, self.place_to_drop = task
        self.task = task
        self.goal = self.obj_to_pick
        print("Env Task: Pick {} and drop it on {}".format(self.obj_to_pick, self.place_to_drop))

    def filter_obs(self, obs):
        # Filter the observations to only include the relevant information
        # If cube1 is the object to pick, then the observation should only include the position and quat of cube1
        # cube1: obs[0:7], cube2: obs[7:14], cube3: obs[14:21] and rest of the obs (21::)
        map_cube_obs = {"cube1": obs[0:7], "cube2": obs[7:14], "cube3": obs[14:21]}
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body][:3])
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        obj_to_pick_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3])
        goal_drop = self.place_to_drop
        #if self.place_to_drop != self.goal:
        #    place_to_drop_pos = np.asarray([0, 0, 0])
        #else:
        place_to_drop_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[goal_drop]][:3])
        if 'peg' in self.place_to_drop:
            place_to_drop_pos = place_to_drop_pos - np.array([0.1, 0.04, 0])
        obs = np.concatenate([gripper_pos, [aperture], place_to_drop_pos, obj_to_pick_pos])

        #print(gripper_pos * 1000)
        #print("XY distance gripper to obj_to_pick: ", np.linalg.norm(gripper_pos[:2] - obj_to_pick_pos[:2]))
        #print("Z gripper: ", gripper_pos[2], " Z obj_to_pick: ", obj_to_pick_pos[2], " Z place_to_drop: ", place_to_drop_pos[2])
        return obs

    def simple_obs(self, obs):
        # Filter the observations to only include the relevant information
        # cube1: obs[0:7], cube2: obs[7:14], cube3: obs[14:21] and rest of the obs (21::)
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body][:3])
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        obs_xyz_cube1 = obs[0:3]
        obs_xyz_cube2 = obs[7:10]
        obs_xyz_cube3 = obs[14:17]
        obs_xyz_peg1 = self.env.sim.data.body_xpos[self.obj_mapping['peg1']][:3] - np.array([0.1, 0.04, 0])
        obs_xyz_peg2 = self.env.sim.data.body_xpos[self.obj_mapping['peg2']][:3] - np.array([0.1, 0.04, 0])
        obs_xyz_peg3 = self.env.sim.data.body_xpos[self.obj_mapping['peg3']][:3] - np.array([0.1, 0.04, 0])
        obs = np.concatenate([gripper_pos, [aperture], obs_xyz_cube1, obs_xyz_cube2, obs_xyz_cube3, obs_xyz_peg1, obs_xyz_peg2, obs_xyz_peg3])

        #print(gripper_pos * 1000)
        #print("XY distance gripper to obj_to_pick: ", np.linalg.norm(gripper_pos[:2] - obj_to_pick_pos[:2]))
        #print("Z gripper: ", gripper_pos[2], " Z obj_to_pick: ", obj_to_pick_pos[2], " Z place_to_drop: ", place_to_drop_pos[2])
        return obs

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
        #print("Displacement action: ", np.linalg.norm(action[:3]))
        # divide the action by 1000 to scale the values
        action = np.asarray(action) / 1000
        # if self.nulified_action_indexes is not empty, fill the action with zeros at the indexes
        if self.nulified_action_indexes != []:
            for index in self.nulified_action_indexes:
                action = np.insert(action, index, 0)
        truncated = False
        action = self.map_gripper(action)
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        self.env.render() if self.render_init else None
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        success = state[f"on({self.obj_to_pick},{self.place_to_drop})"] and not(state[f"grasped({self.obj_to_pick})"])
        info['is_success'] = success
        #if success:
        #    print(f"Object {self.obj_to_pick} successfully placed on the drop-off location {self.place_to_drop}", state[f"on({self.obj_to_pick},{self.place_to_drop})"])
        truncated = truncated or self.env.done
        terminated = (terminated or success) if not(self.hanoi) else terminated
        if state[f"on({self.obj_to_pick},{self.place_to_drop})"] or state['grasped({})'.format(self.obj_to_pick)]:
            #obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]][:3]))
            self.goal = self.place_to_drop
        else:
            #obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3]))
            self.goal = self.obj_to_pick
        #print("ENV Goal: ", self.goal)
        if state[f"over(gripper,{self.obj_to_pick})"]:
            reward = 0.25
        elif state[f"grasped({self.obj_to_pick})"]:
            reward = 0.5
        elif state[f"over(gripper,{self.place_to_drop})"]:
            reward = 0.75
        elif success:
            reward = 1
        else:
            reward = 0
        self.step_count += 1
        if self.step_count > self.horizon:
            print("Horizon reached within environment")
            terminated = True
        if self.oracle:
            obs_copy = copy.deepcopy(obs)
            obs = self.filter_obs(obs)
            obs_copy = self.simple_obs(obs_copy)
            info["obs_base"] = obs_copy
            info["agentview"] = self.env.env._get_observations()["agentview_image"]
            info["robot0_eye_in_hand"] = self.env.env._get_observations()["robot0_eye_in_hand_image"]
            xyz_cube1 = self.env.sim.data.body_xpos[self.obj_mapping["cube1"]][:3]
            xyz_cube2 = self.env.sim.data.body_xpos[self.obj_mapping["cube2"]][:3]
            xyz_cube3 = self.env.sim.data.body_xpos[self.obj_mapping["cube3"]][:3]
            if "cube4" in self.obj_mapping:
                xyz_cube4 = self.env.sim.data.body_xpos[self.obj_mapping["cube4"]][:3]
                info["cubes_obs"] = {"cube1": xyz_cube1, "cube2": xyz_cube2, "cube3": xyz_cube3, "cube4": xyz_cube4}
            info["cubes_obs"] = {"cube1": xyz_cube1, "cube2": xyz_cube2, "cube3": xyz_cube3}
            # Add ee position to the info
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body][:3])
            info["ee_pos"] = gripper_pos
        else:
            obs_filter = copy.deepcopy(obs)
            obs = self.simple_obs(obs)
            obs_filter = self.filter_obs(obs_filter)
            info["obs_filter"] = obs_filter
        # x1000 to scale the values
        obs = obs * 1000
        goal_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3]
        self.keypoint = goal_pos
        info["keypoint"] = self.keypoint
        info["state"] = state
        info["task"] = (self.obj_to_pick, self.place_to_drop)
        return obs, reward, terminated, truncated, info
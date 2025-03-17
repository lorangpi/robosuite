import copy
import gymnasium as gym
#import gym
import robosuite as suite
import numpy as np
from robosuite.utils.detector import HanoiDetector

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

class HanoiPlaceWrapper(gym.Wrapper):
    def __init__(self, env, render_init=False, nulified_action_indexes=[], horizon=200, goal_type='random', image_obs=True):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.render_init = render_init
        self.detector = HanoiDetector(self)
        self.nulified_action_indexes = nulified_action_indexes
        self.horizon = horizon
        self.step_count = 1
        self.goal_type = goal_type
        self.image_obs = image_obs

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

        # set up observation space
        self.obs_dim = self.env.obs_dim
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
        #print("Task: Pick {} and drop it on {}".format(self.obj_to_pick, self.place_to_drop))
        return f"on({cube_to_pick},{place_to_drop})"

    def reach_drop_reset(self):
        """
        Resets the environment to a state where the gripper is holding the object on top of the drop-off location
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state

        self.reset_step_count = 0

        # # Moving randomly 0 to 50 steps
        # for k in range(np.random.randint(1, 50)):
        #     generate_random_3d_action = np.random.uniform(-0.2, 0.2, 3)
        #     action = np.concatenate([generate_random_3d_action, [0]])
        #     obs,_,_,_,_ = self.env.step(action)
        #     self.env.render() if self.render_init else None

        #print("Moving up...")
        for _ in range(5):
            obs,_,_,_,_ = self.env.step([0,0,1,0])
            self.env.render() if self.render_init else None
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state

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

        return True, obs

    def valid_state(self):
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        state = {k: state[k] for k in state if 'on' in k}
        # Filter only the values that are True
        state = {key: value for key, value in state.items() if value}
        # if state has not 3 keys, return None
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
                success, obs = self.reach_drop_reset()
                if trials > 3:
                    break   
            if self.goal_type != 'random':
                success = False
                for trial in range(20):
                    self.task = self.sample_task()
                    if self.goal_type in self.place_to_drop:
                        success = True
                        break
            else:
                self.task = self.sample_task()
            reset = success

        self.sim.forward()
        self.goal = self.place_to_drop
        #obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]][:3]))
        if not self.image_obs:
            obs = self.filter_obs(obs)
        goal_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.goal]][:3]
        goal_quat = self.env.sim.data.body_xquat[self.obj_mapping[self.goal]]

        self.keypoint = np.concatenate([goal_pos, goal_quat])
        info["keypoint"] = self.keypoint
        info["state"] = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.success_steps = 0

        return obs, info

    def filter_obs_proprio(self, obs):
        # Filter the observations to only include the relevant information
        # If cube1 is the object to pick, then the observation should only include the position and quat of cube1
        # cube1: obs[0:7], cube2: obs[7:14], cube3: obs[14:21] and rest of the obs (21::)
        map_cube_obs = {"cube1": obs[0:7], "cube2": obs[7:14], "cube3": obs[14:21]}
        if 'cube' in self.goal:
            return np.concatenate([map_cube_obs[self.goal], obs[21:]])
        elif 'peg' in self.goal:
            peg_pos = self.env.env.sim.data.body_xpos[self.obj_mapping[self.goal]][:3]
            peg_pos = np.concatenate([peg_pos, [0, 0, 0, 1]])
            return np.concatenate([peg_pos, obs[21:]])
        
    def filter_obs(self, obs):
        # Filter the observations to only include the relevant information
        # If cube1 is the object to pick, then the observation should only include the position and quat of cube1
        # cube1: obs[0:7], cube2: obs[7:14], cube3: obs[14:21] and rest of the obs (21::)
        map_cube_obs = {"cube1": obs[0:7], "cube2": obs[7:14], "cube3": obs[14:21]}
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body][:3])
        gripper_quat = np.asarray(self.env.sim.data.body_xquat[self.env.gripper_body])
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        if 'cube' in self.goal:
            #return np.concatenate([map_cube_obs[self.goal], obs[21:]])
            return np.concatenate([map_cube_obs[self.goal], gripper_pos, gripper_quat, [aperture]])
        elif 'peg' in self.goal:
            peg_pos = self.env.env.sim.data.body_xpos[self.obj_mapping[self.goal]][:3] - np.array([0.1, 0.04, 0])
            peg_pos = np.concatenate([peg_pos, [0, 0, 0, 1]])
            #return np.concatenate([peg_pos, obs[21:]])
            return np.concatenate([peg_pos, gripper_pos, gripper_quat, [aperture]])

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
        # if self.nulified_action_indexes is not empty, fill the action with zeros at the indexes
        if self.nulified_action_indexes != []:
            for index in self.nulified_action_indexes:
                action = np.insert(action, index, 0)
        action = self.map_gripper(action)
        truncated = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        terminated = bool(terminated)
        info["is_success"] = False
        self.env.render() if self.render_init else None
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        
        # Get reward
        reward = self.staged_rewards(state)

        # Check if the object has been successfully picked up
        success = state[f"on({self.obj_to_pick},{self.place_to_drop})"] and not(state[f"grasped({self.obj_to_pick})"])
        if success:
            self.success_steps += 1
            reward = 1000 - self.step_count*5
            if self.success_steps >= 5:  # Require 5 steps of stability
                print("Object successfully placed", state[f"on({self.obj_to_pick},{self.place_to_drop})"])
                info['is_success'] = True
                terminated = True
                reward = 2000 - self.step_count*10
        
        truncated = truncated or self.env.done

        self.step_count += 1
        if self.step_count > self.horizon:
            terminated = True

        if not self.image_obs:
            obs = self.filter_obs(obs)
        
        info["keypoint"] = self.keypoint
        info["state"] = state

        return obs, reward, terminated, truncated, info

    def staged_rewards(self, state):
        distances = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=True)

        MAX_APPROACH_DIST = 0.5
        MAX_DROP_DIST = 0.2
        MAX_PICKED_DIST = 0.1

        reward = 0  # Neutral baseline

        # *** Stage 1: Success (Final Goal - Object Placed) ***
        if state[f"on({self.obj_to_pick},{self.place_to_drop})"]:
            drop_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]][2]
            object_z_loc = self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][2]
            z_dist = np.abs(drop_pos - object_z_loc)
            reward = 100 + 100 * (1.0 - np.clip(z_dist / MAX_PICKED_DIST, 0, 1))  # Bonus for precise placement

        # *** Stage 2: Gripper Over Target and at Drop Level ***
        elif state[f"over(gripper,{self.place_to_drop})"]:
            drop_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.detector.object_id[self.place_to_drop])][2]
            gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][2]
            drop_level_dist = np.linalg.norm(drop_pos - gripper_pos)
            reward = 50 + 50 * (1.0 - np.clip(drop_level_dist / MAX_DROP_DIST, 0, 1))  # Reward reaching drop level

            if state[f"grasped({self.obj_to_pick})"]:
                reward += 20  # Encourage holding the object before releasing

        # *** Stage 3: Moving Object to Drop Position ***
        elif state[f"picked_up({self.obj_to_pick})"]:
            drop_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]][:2]
            gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:2]
            dist = np.linalg.norm(drop_pos - gripper_pos)
            reward = 10 + 10 * (1.0 - np.clip(dist / MAX_APPROACH_DIST, 0, 1))  # Reward approaching target location

        # *** Stage 4: Still Holding Object but Not Moving Toward Goal ***
        elif state[f"grasped({self.obj_to_pick})"]:
            dist = distances[f"picked_up({self.obj_to_pick})"]
            reward = np.clip(dist / MAX_PICKED_DIST, 0, 1)  # Penalize not approaching drop location
        else:
            pick_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:2]
            gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:2]
            dist = np.linalg.norm(pick_pos - gripper_pos)
            reward = - 10 * np.clip(dist / MAX_APPROACH_DIST, 0, 1)  # Penalize not approaching drop location

        return reward

        # MAX_APPROACH_DIST = 0.5   # maximum distance for approaching the object
        # MAX_GRAB_DIST = 0.2       # maximum distance considered for grab-level alignment
        # MAX_PICKED_DIST = 0.1     # maximum distance for the picked-up stage

        # if success:
        #     reward = 1000
        # elif state[f"on({self.obj_to_pick},{self.place_to_drop})"]:
        #     reward = 3
        # elif state[f"over(gripper,{self.place_to_drop})"]:
        #     drop_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]][2]
        #     gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][2]
        #     dist = np.linalg.norm(drop_pos - gripper_pos)
        #     reward = 2 - np.clip(dist/MAX_GRAB_DIST, 0, 1)#(np.tanh(20 * dist))
        # elif state[f"picked_up({self.obj_to_pick})"]:
        #     drop_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]][:2]
        #     gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:2]
        #     dist = np.linalg.norm(drop_pos - gripper_pos)
        #     reward = 1 - np.clip(dist/MAX_APPROACH_DIST, 0, 1)#(np.tanh(10 * dist))
        # else:
        #     pick_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][2]
        #     table_pos = self.env.table_offset[2] + 0.45
        #     dist = np.linalg.norm(pick_pos - table_pos)
        #     reward = - np.clip(dist/MAX_PICKED_DIST, 0, 1) #-8 - (np.tanh(20 * dist))
        # if not(state[f"grasped({self.obj_to_pick})"]):
        #     pick_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3]
        #     gripper_pos = self.env.sim.data.body_xpos[self.gripper_body][:3]
        #     dist = np.linalg.norm(pick_pos - gripper_pos)
        #     reward = -20 - np.clip(dist/MAX_APPROACH_DIST, 0, 1)#- (np.tanh(5 * dist))
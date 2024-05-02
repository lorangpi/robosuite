import copy
import gymnasium as gym
import robosuite as suite
import numpy as np
from detector import Robosuite_Hanoi_Detector
from PDDL.executor import Executor_RL

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

class PoliciesResetWrapper(gym.Wrapper):
    def __init__(self, env, render_init=False, nulified_action_indexes=[], horizon=500, prev_action_policies:list=[Executor_RL]):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.use_gripper = True
        self.render_init = render_init
        self.detector = Robosuite_Hanoi_Detector(self)
        self.nulified_action_indexes = nulified_action_indexes
        self.horizon = horizon
        self.prev_action_policies = prev_action_policies
        self.step_count = 1

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
        self.obj_name2body_mapping = {'cube1': self.cube1_body, 'cube2': self.cube2_body, 'cube3': self.cube3_body, 'peg1': self.peg1_body, 'peg2': self.peg2_body, 'peg3': self.peg3_body}
        
        self.goal_obj_name2number_mapping = {'cube1': 0, 'cube2': 1, 'cube3': 2, 'peg1': 3, 'peg2': 4, 'peg3': 5}
        self.area_pos = {'peg1': self.env.pegs_xy_center[0], 'peg2': self.env.pegs_xy_center[1], 'peg3': self.env.pegs_xy_center[2]}

        # set up observation space
        self.obs_dim = self.env.obs_dim + 3 # 1 extra dimensions for the object goal

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)
        self.action_space = gym.spaces.Box(low=self.env.action_space.low[:-len(nulified_action_indexes)], high=self.env.action_space.high[:-len(nulified_action_indexes)], dtype=np.float64)

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

    def reset_using_action_policies(self, obs):
        """Resets the environment by executing the policies in `self.prev_action_policies` in sequence. Each policy corresponds to a high level action that needs to be completed.

        Args:
            obs (ObsType): an observation of the initial state
        """
        def get_action_step_goals():
            """Finds the symbolic goal and the (x, y, z) position
            goal for the current action step
            Returns:
                goal: the location of the object to be picked up or the drop location
                symgoal: name of the object to be picked up or the name of the drop location 
            """
            if 'Pick' in prev_action_policy.id:
                goal = self.env.sim.data.body_xpos[self.obj_name2body_mapping[self.obj_to_pick]][:3]
                symgoal = self.obj_to_pick
            elif 'Drop' in prev_action_policy.id:
                if 'peg' in self.place_to_drop:
                    drop_loc = self.area_pos[self.place_to_drop]
                else:
                    drop_loc = self.env.sim.data.body_xpos[self.obj_name2body_mapping[self.place_to_drop]][:3]
                goal = drop_loc
                if 'Reach' in prev_action_policy.id:
                    symgoal = self.place_to_drop
                else:
                    symgoal = (self.obj_to_pick,self.place_to_drop)
            return goal, symgoal
            
            
        for prev_action_policy in self.prev_action_policies:
            goal, symgoal = get_action_step_goals()
            obs, success = prev_action_policy.execute(self.env.env, obs, goal, symgoal, render=False)
            
            if not success or not self.valid_state():
                return False, obs

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
            self.task = self.sample_task()
            self.env.reset_state = self.reset_state
            success = False
            while not success:
                valid_state = False
                while not valid_state:
                    #print("Trying to reset the environment...")
                    if len(self.prev_action_policies) == 0: # if no policies for previous actions, just use this current action's reset function, which is wrapped by the current PoliciesResetWrapper
                        try:
                            obs, info = self.env.reset()
                        except:
                            obs = self.env.reset()
                            info = {}
                    else: # otherwise we need to call the reset function of the original env without any wrappers
                        try:
                            obs, info = self.env.env.reset()
                        except:
                            obs = self.env.env.reset()
                            info = {}
                    valid_state = self.valid_state()
                    trials += 1
                    if trials > 3:
                        break   
                success, obs = self.reset_using_action_policies(obs=obs)
                reset = success
                if trials > 3:
                    break   

        self.sim.forward()
        # replace the goal object id with its array of x, y, z location
        obs = np.concatenate((obs, self.env.sim.data.body_xpos[self.obj_name2body_mapping[self.place_to_drop]][:3]))
        return obs, info

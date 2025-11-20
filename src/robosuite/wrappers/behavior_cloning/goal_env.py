import gymnasium as gym
import numpy as np

class GoalEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Desired goal is the (x, y, z) of the self.env.goal object and the Achieved goal is the (x, y, z) of the gripper  
        goal_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'observation': self.env.observation_space,
            'achieved_goal': goal_space,
            'desired_goal': goal_space
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        achieved_goal = self.env.sim.data.body_xpos[self.gripper_body][:3]
        desired_goal = self.env.goal
        return {'observation': obs, 'achieved_goal': achieved_goal, 'desired_goal': desired_goal}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        achieved_goal = self.env.sim.data.body_xpos[self.gripper_body][:3]
        desired_goal = self.env.goal
        return {'observation': obs, 'achieved_goal': achieved_goal, 'desired_goal': desired_goal}, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, terminated, truncated, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        return -np.linalg.norm(achieved_goal - desired_goal, axis=-1)
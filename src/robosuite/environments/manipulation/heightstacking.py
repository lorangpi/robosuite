from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

class HeightStacking(SingleArmEnv):
    """
    This class corresponds to a height stacking task for a single robot arm.
    A random number of cubes (1-5) with different sizes are placed on a line,
    and must be stacked on a central platform to maximize height.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        min_cubes (int): Minimum number of cubes to spawn (default: 1)
        
        max_cubes (int): Maximum number of cubes to spawn (default: 5)

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        min_cubes=4,
        max_cubes=4,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # cube settings
        self.min_cubes = min_cubes
        self.max_cubes = max_cubes
        self.num_cubes = max_cubes  # Initialize with max, will be set during reset
        
        # Define possible cube sizes (5 different sizes)
        self.cube_sizes_options = [0.017, 0.019, 0.021, 0.023, 0.025]
        
        # Define distinct colors for different sized cubes
        self.cube_colors = [
            [0, 0, 1, 1],      # Blue
            [1, 0, 0, 1],      # Red
            [0, 1, 0, 1],      # Green
            [1, 1, 0, 1],      # Yellow
            [1, 0, 1, 1],      # Magenta
        ]

        self.rgba_semantic_colors = {
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1],
            "blue": [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1],
        }

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        
        # Platform position (center of table)
        self.platform_pos = np.array([0.00, 0.24, self.table_offset[2] + 0.005])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action):
        """
        Reward function for the task.

        The reward encourages:
        - Placing all cubes on the platform
        - Maximizing the total height of the stack
        - Stable stacking (cubes aligned on top of each other)

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0
        
        # Get positions of all active cubes
        platform_pos_2d = self.platform_pos[:2]
        cubes_on_platform = []
        cubes_heights = []
        
        for i in range(self.active_num_cubes):
            cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
            
            # Check if cube is on platform
            on_platform = self._check_on_platform(cube_pos, self.platform_pos)
            
            if on_platform:
                cubes_on_platform.append(i)
                cubes_heights.append(cube_pos[2])
        
        if self.reward_shaping:
            # Dense reward components
            for i in range(self.active_num_cubes):
                cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
                gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
                
                # Check if cube is on platform
                on_platform = self._check_on_platform(cube_pos, self.platform_pos)
                
                if on_platform:
                    # Reward for having cube on platform
                    reward += 1.0
                    
                    # Bonus reward for height (higher is better)
                    height_above_platform = cube_pos[2] - self.platform_pos[2]
                    reward += 0.5 * height_above_platform / 0.1  # Normalize by expected max height
                else:
                    # Small reward for approaching platform with cube
                    grasping = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubes[i])
                    if grasping:
                        dist_to_platform = np.linalg.norm(cube_pos[:2] - platform_pos_2d)
                        reward += 0.5 * (1 - np.tanh(2.0 * dist_to_platform))
                    else:
                        # Tiny reward for approaching cube
                        dist_to_cube = np.linalg.norm(gripper_site_pos - cube_pos)
                        reward += 0.1 * (1 - np.tanh(10.0 * dist_to_cube))
        else:
            # Sparse reward: count cubes on platform + height bonus
            num_on_platform = len(cubes_on_platform)
            reward = float(num_on_platform)
            
            # Bonus for total stack height
            if num_on_platform > 0:
                max_height = max(cubes_heights)
                height_bonus = (max_height - self.platform_pos[2]) / 0.05  # Normalize
                reward += height_bonus

        if self.reward_scale is not None:
            reward *= self.reward_scale / (self.active_num_cubes + 2)  # +2 for height bonus

        return reward

    def _check_on_platform(self, cube_pos, platform_pos):
        """
        Check if a cube is on the platform.
        
        Args:
            cube_pos: Position of the cube
            platform_pos: Position of the platform center
            
        Returns:
            bool: True if cube is on platform
        """
        # Platform dimensions (12cm x 12cm)
        platform_half_size = 0.06
        
        # Check if cube is within platform boundaries
        x_on = abs(cube_pos[0] - platform_pos[0]) < platform_half_size
        y_on = abs(cube_pos[1] - platform_pos[1]) < platform_half_size
        z_on = cube_pos[2] > platform_pos[2] and cube_pos[2] < platform_pos[2] + 0.05  # Within reasonable height
        
        return x_on and y_on and z_on

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }

        text_number1 = CustomMaterial(
            texture="Number1",
            tex_name="number1",
            mat_name="number1_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        text_number2 = CustomMaterial(
            texture="Number2",
            tex_name="number2",
            mat_name="number2_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        text_number3 = CustomMaterial(
            texture="Number3",
            tex_name="number3",
            mat_name="number3_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        text_number4 = CustomMaterial(
            texture="Number4",
            tex_name="number4",
            mat_name="number4_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        map_textures = {"blue": text_number1, "red": text_number2, "green": text_number3, "yellow": text_number4}

        # Create maximum number of cubes (will activate subset during reset)
        self.cubes = []
        self.cube_sizes = []
        self.cube_body_ids = []
        
        for i in range(self.max_cubes):
            # Create cube with size from options
            size = self.cube_sizes_options[i]
            rgba = self.cube_colors[i]
            for key in self.rgba_semantic_colors:
                if rgba == self.rgba_semantic_colors[key]:
                    material = map_textures[key]
                    break
            
            cube = BoxObject(
                name=f"cube{i}",
                size_min=[size, size, size],
                size_max=[size, size, size],
                rgba=rgba,
                material=material,
            )
            
            self.cubes.append(cube)
            self.cube_sizes.append(size)

        # Create stacking platform (visual marker at center)
        self.platform = BoxObject(
            name="platform",
            size_min=[0.06, 0.06, 0.0001],  # 12cm x 12cm x 0.2cm
            size_max=[0.06, 0.06, 0.0001],
            rgba=[0.5, 0.5, 0.5, 1],  # Gray platform
            obj_type="visual",
            joints=None,
        )
        
        # Initial line position for cubes (x=0, along y-axis)
        self.line_x = 0.00
        self.line_y_start = -0.23
        self.line_y_end = 0.12
        
        self.objects = self.cubes + [self.platform]

        # Create placement initializer for platform
        self.platform_placement_initializer = UniformRandomSampler(
            name="PlatformSampler",
            mujoco_objects=self.platform,
            x_range=[self.platform_pos[0], self.platform_pos[0]],
            y_range=[self.platform_pos[1], self.platform_pos[1]],
            rotation=0,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=self.table_offset,
            z_offset=0.005,
        )

        # Create placement initializers for cubes along the line
        self.cube_placement_initializers = []
        
        for i, cube in enumerate(self.cubes):
            sampler = UniformRandomSampler(
                name=f"Cube{i}Sampler",
                mujoco_objects=cube,
                x_range=[self.line_x, self.line_x],
                y_range=[self.line_y_start, self.line_y_end],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            self.cube_placement_initializers.append(sampler)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_ids = []
        for cube in self.cubes:
            self.cube_body_ids.append(self.sim.model.body_name2id(cube.root_body))

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.gripper_body = self.sim.model.body_name2id('gripper0_eef')
        # Randomly choose number of cubes for this episode
        self.active_num_cubes = np.random.randint(self.min_cubes, self.max_cubes + 1)
        
        # Randomly select which cube sizes to use (without replacement)
        active_cube_indices = np.random.choice(self.max_cubes, self.active_num_cubes, replace=False)
        active_cube_indices = sorted(active_cube_indices)  # Sort for consistency
        
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for platform
            platform_placement = self.platform_placement_initializer.sample()
            for obj_pos, obj_quat, obj in platform_placement.values():
                body_id = self.sim.model.body_name2id(obj.root_body)
                self.sim.model.body_pos[body_id] = obj_pos
                self.sim.model.body_quat[body_id] = obj_quat

            # Randomly assign positions along line for active cubes
            y_positions = np.linspace(self.line_y_start, self.line_y_end, self.active_num_cubes)
            np.random.shuffle(y_positions)
            
            # Place active cubes on the line
            for idx, cube_idx in enumerate(active_cube_indices):
                cube = self.cubes[cube_idx]
                sampler = self.cube_placement_initializers[cube_idx]
                
                # Update sampler with assigned position
                sampler.y_range = [y_positions[idx], y_positions[idx]]
                cube_placement = sampler.sample()
                
                # Set cube position
                for obj_pos, obj_quat, obj in cube_placement.values():
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
            
            # Move inactive cubes far away (below table, invisible)
            for cube_idx in range(self.max_cubes):
                if cube_idx not in active_cube_indices:
                    cube = self.cubes[cube_idx]
                    # Move cube far below the table
                    far_away_pos = np.array([0, 0, -10.0])
                    far_away_quat = np.array([1, 0, 0, 0])
                    self.sim.data.set_joint_qpos(cube.joints[0], np.concatenate([far_away_pos, far_away_quat]))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # Create observables for each cube
            for i in range(self.max_cubes):
                
                @sensor(modality=modality)
                def cube_pos(obs_cache, cube_idx=i):
                    return np.array(self.sim.data.body_xpos[self.cube_body_ids[cube_idx]])

                @sensor(modality=modality)
                def cube_quat(obs_cache, cube_idx=i):
                    return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_ids[cube_idx]]), to="xyzw")

                @sensor(modality=modality)
                def gripper_to_cube(obs_cache, cube_idx=i):
                    cube_pos_key = f"cube{cube_idx}_pos"
                    return (
                        obs_cache[cube_pos_key] - obs_cache[f"{pf}eef_pos"]
                        if cube_pos_key in obs_cache and f"{pf}eef_pos" in obs_cache
                        else np.zeros(3)
                    )

                @sensor(modality=modality)
                def cube_to_platform(obs_cache, cube_idx=i):
                    cube_pos_key = f"cube{cube_idx}_pos"
                    if cube_pos_key in obs_cache:
                        return self.platform_pos - obs_cache[cube_pos_key]
                    return np.zeros(3)

                # Add observables
                observables[f"cube{i}_pos"] = Observable(
                    name=f"cube{i}_pos",
                    sensor=cube_pos,
                    sampling_rate=self.control_freq,
                )
                
                observables[f"cube{i}_quat"] = Observable(
                    name=f"cube{i}_quat",
                    sensor=cube_quat,
                    sampling_rate=self.control_freq,
                )
                
                observables[f"gripper_to_cube{i}"] = Observable(
                    name=f"gripper_to_cube{i}",
                    sensor=gripper_to_cube,
                    sampling_rate=self.control_freq,
                )
                
                observables[f"cube{i}_to_platform"] = Observable(
                    name=f"cube{i}_to_platform",
                    sensor=cube_to_platform,
                    sampling_rate=self.control_freq,
                )

            # Add platform position observable
            @sensor(modality=modality)
            def platform_pos_obs(obs_cache):
                return self.platform_pos

            observables["platform_pos"] = Observable(
                name="platform_pos",
                sensor=platform_pos_obs,
                sampling_rate=self.control_freq,
            )

        return observables

    def _check_success(self):
        """
        Check if all active cubes are stacked on the platform.

        Returns:
            bool: True if all cubes are on platform
        """
        for i in range(self.active_num_cubes):
            cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
            
            if not self._check_on_platform(cube_pos, self.platform_pos):
                return False
                
        return True

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the nearest cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the nearest active cube
        if vis_settings["grippers"]:
            # Find nearest active cube not on platform
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            min_dist = float('inf')
            nearest_cube = None
            
            for i in range(self.active_num_cubes):
                cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
                
                # Only consider cubes not yet on platform
                if not self._check_on_platform(cube_pos, self.platform_pos):
                    dist = np.linalg.norm(gripper_site_pos - cube_pos)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_cube = self.cubes[i]
            
            if nearest_cube is not None:
                self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=nearest_cube)
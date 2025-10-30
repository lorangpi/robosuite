from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.objects import PlateVisualObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat


class CubeSorting(SingleArmEnv):
    """
    This class corresponds to a cube sorting task for a single robot arm.
    Multiple cubes of two different sizes are placed on a line and should be
    sorted onto two different platforms by size.

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

        num_cubes (int): Number of cubes to spawn (default: 6)

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
        num_cubes=4,
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
        self.num_cubes = num_cubes
        self.small_size = 0.018
        self.large_size = 0.022

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

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
        - Small cubes on platform 1
        - Large cubes on platform 2

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0
        
        # Check each cube
        for i, (cube, size) in enumerate(zip(self.cubes, self.cube_sizes)):
            cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
            
            # Check if cube is on platform 1 (small cubes platform)
            on_platform1 = self._check_on_platform(cube_pos, self.platform1_pos)
            # Check if cube is on platform 2 (large cubes platform)
            on_platform2 = self._check_on_platform(cube_pos, self.platform2_pos)
            
            # Reward for correct placement
            if size == "small" and on_platform1:
                reward += 1.0
            elif size == "large" and on_platform2:
                reward += 1.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / self.num_cubes

        return reward

    def _check_on_platform(self, cube_pos, platform_pos):
        """
        Check if a cube is on a platform.
        
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
        z_on = cube_pos[2] > self.table_offset[2] #+ 0.005  # Just above the platform
        z_on = z_on and cube_pos[2] < (self.table_offset[2] + 0.05)  # Below a certain height
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

        # Create cubes with random sizes
        self.cubes = []
        self.cube_sizes = []
        self.cube_body_ids = []

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

        map_textures = {"small": text_number1, "large": text_number2}

        # Randomly assign sizes to cubes (approximately half small, half large)
        for i in range(self.num_cubes):
            size_type = "small" if np.random.rand() < 0.5 else "large"
            size = self.small_size if size_type == "small" else self.large_size
            
            # Color code: blue for small, red for large
            rgba = [0, 0, 1, 1] if size_type == "small" else [1, 0, 0, 1]
            color = "blue" if size_type == "small" else "red"
            
            cube = BoxObject(
                name=f"cube{i}",
                size_min=[size, size, size],
                size_max=[size, size, size],
                rgba=rgba,
                material=map_textures[size_type],
            )
            
            self.cubes.append(cube)
            self.cube_sizes.append(size_type)

        # Create two platforms with colors matching cube sizes
        # Platform 1: Blue (for small/blue cubes)
        # Platform 2: Red (for large/red cubes)
        self.platform1 = BoxObject(
            name="platform1",
            size_min=[0.06, 0.06, 0.0001],  # Flat platform: 12cm x 12cm x 1cm
            size_max=[0.06, 0.06, 0.0001],
            rgba=[0, 0, 1, 1],  # Blue for small cubes
            obj_type="visual",
            joints=None,
        )

        self.platform2 = BoxObject(
            name="platform2",
            size_min=[0.06, 0.06, 0.0001],  # Flat platform: 12cm x 12cm x 1cm
            size_max=[0.06, 0.06, 0.0001],
            rgba=[1, 0, 0, 1],  # Red for large cubes
            obj_type="visual",
            joints=None,
        )

        # All objects on the same line (y-axis)
        self.line_x = 0.05
        
        # Platform positions (on the same line as cubes)
        self.platform1_pos = np.array([self.line_x, -0.26, self.table_offset[2] + 0.005])
        self.platform2_pos = np.array([self.line_x, 0.26, self.table_offset[2] + 0.005])
        
        # Cube placement range (between the two platforms)
        self.line_y_start = -0.17
        self.line_y_end = 0.17
        
        self.objects = self.cubes + [self.platform1, self.platform2]

        # Create placement initializers for platforms
        self.platform_placement_initializer = SequentialCompositeSampler(name="PlatformSampler")
        
        # Platform 1 placement
        self.platform_placement_initializer.append_sampler(
            UniformRandomSampler(
                name="Platform1Sampler",
                mujoco_objects=self.platform1,
                x_range=[self.platform1_pos[0], self.platform1_pos[0]],
                y_range=[self.platform1_pos[1], self.platform1_pos[1]],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.005,  # Slightly above table
            ))
        
        # Platform 2 placement
        self.platform_placement_initializer.append_sampler(
            UniformRandomSampler(
                name="Platform2Sampler",
                mujoco_objects=self.platform2,
                x_range=[self.platform2_pos[0], self.platform2_pos[0]],
                y_range=[self.platform2_pos[1], self.platform2_pos[1]],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.005,  # Slightly above table
            ))

        # Create placement initializers for cubes along the line
        self.cube_placement_initializers = []
        y_positions = np.linspace(self.line_y_start, self.line_y_end, self.num_cubes)
        
        for i, cube in enumerate(self.cubes):
            sampler = UniformRandomSampler(
                name=f"Cube{i}Sampler",
                mujoco_objects=cube,
                x_range=[self.line_x, self.line_x],
                y_range=[y_positions[i], y_positions[i]],
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

        # Randomly reassign cube sizes on each reset
        # self.cube_sizes = []
        # for i in range(self.num_cubes):
        #     size_type = "small" if np.random.rand() < 0.5 else "large"
        #     self.cube_sizes.append(size_type)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for platforms
            platform_placements = self.platform_placement_initializer.sample()
            for obj_pos, obj_quat, obj in platform_placements.values():
                body_id = self.sim.model.body_name2id(obj.root_body)
                self.sim.model.body_pos[body_id] = obj_pos
                self.sim.model.body_quat[body_id] = obj_quat

            # Randomly shuffle the y-positions for cube placement
            y_positions = np.linspace(self.line_y_start, self.line_y_end, self.num_cubes)
            np.random.shuffle(y_positions)
            
            # Sample from the placement initializer for cubes
            for i, (cube, sampler) in enumerate(zip(self.cubes, self.cube_placement_initializers)):
                # Update sampler with shuffled position
                sampler.y_range = [y_positions[i], y_positions[i]]
                cube_placement = sampler.sample()
                
                # Set cube position
                for obj_pos, obj_quat, obj in cube_placement.values():
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

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
            for i, cube in enumerate(self.cubes):
                
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

        return observables

    def _check_success(self):
        """
        Check if all cubes are correctly sorted on platforms.

        Returns:
            bool: True if all cubes are correctly sorted
        """
        for i, (cube, size) in enumerate(zip(self.cubes, self.cube_sizes)):
            cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
            
            on_platform1 = self._check_on_platform(cube_pos, self.platform1_pos)
            on_platform2 = self._check_on_platform(cube_pos, self.platform2_pos)
            
            # Check if cube is on correct platform
            if size == "small" and not on_platform1:
                return False
            elif size == "large" and not on_platform2:
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

        # Color the gripper visualization site according to its distance to the nearest cube
        if vis_settings["grippers"]:
            # Find nearest cube
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            min_dist = float('inf')
            nearest_cube = None
            
            for i, cube in enumerate(self.cubes):
                cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
                dist = np.linalg.norm(gripper_site_pos - cube_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cube = cube
            
            if nearest_cube is not None:
                self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=nearest_cube)
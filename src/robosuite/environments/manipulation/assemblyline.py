from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat


class AssemblyLineSorting(SingleArmEnv):
    """
    This class corresponds to an assembly line sorting task for a single robot arm.
    Cubes arrive on a conveyor-like line and must be sorted to appropriate bins
    based on their color/size.

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

        num_cubes (int): Number of cubes on the assembly line (default: 4)

        num_bins (int): Number of sorting bins (default: 3)

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
        num_bins=3,
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

        # task settings
        self.num_cubes = num_cubes
        self.num_bins = num_bins
        self.cube_size = 0.02

        # Define color categories (RGB colors)
        self.color_categories = [
            ("red", [1, 0, 0, 1]),
            ("green", [0, 1, 0, 1]),
            ("blue", [0, 0, 1, 1]),
        ]

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
        - Placing each cube in its correct bin based on color
        - Dense rewards for grasping and approaching correct bins

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0
        
        if self.reward_shaping:
            # Dense reward components
            for i in range(self.num_cubes):
                cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
                target_bin_idx = self.cube_colors[i]
                target_bin_pos = self.bin_positions[target_bin_idx]
                
                # Reward for grasping
                gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
                dist_to_cube = np.linalg.norm(gripper_site_pos - cube_pos)
                
                grasping = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubes[i])
                if grasping:
                    reward += 0.25
                    
                    # Reward for moving toward correct bin while grasping
                    dist_to_bin = np.linalg.norm(cube_pos[:2] - target_bin_pos[:2])
                    reward += 0.25 * (1 - np.tanh(2.0 * dist_to_bin))
                else:
                    # Small reward for approaching cube
                    reward += 0.1 * (1 - np.tanh(10.0 * dist_to_cube))
                
                # Check if correctly placed
                if self._check_in_bin(cube_pos, target_bin_pos):
                    reward += 1.0
        else:
            # Sparse reward: count correctly sorted cubes
            for i in range(self.num_cubes):
                cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
                target_bin_idx = self.cube_colors[i]
                target_bin_pos = self.bin_positions[target_bin_idx]
                
                if self._check_in_bin(cube_pos, target_bin_pos):
                    reward += 1.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / self.num_cubes

        return reward

    def _check_in_bin(self, cube_pos, bin_pos):
        """
        Check if a cube is correctly placed in a bin.
        
        Args:
            cube_pos: Position of the cube
            bin_pos: Position of the bin center
            
        Returns:
            bool: True if cube is in bin
        """
        # Bin dimensions (12cm x 12cm, slightly smaller check area)
        bin_half_size = 0.05
        
        # Check if cube is within bin boundaries
        x_in = abs(cube_pos[0] - bin_pos[0]) < bin_half_size
        y_in = abs(cube_pos[1] - bin_pos[1]) < bin_half_size
        z_above_bin = cube_pos[2] > self.table_offset[2] + 0.005  # Just above the bin platform
        
        return x_in and y_in and z_above_bin

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

        # Create cubes with colors
        self.cubes = []
        self.cube_colors = []  # Store color category index for each cube
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

        map_textures = {"blue": text_number1, "red": text_number2, "green": text_number3, "yellow": text_number4}
        
        for i in range(self.num_cubes):
            # Randomly assign color category (will be set during reset)
            color_idx = i % self.num_bins  # Initial assignment
            color_name, rgba = self.color_categories[color_idx]
            
            cube = BoxObject(
                name=f"cube{i}",
                size_min=[self.cube_size, self.cube_size, self.cube_size],
                size_max=[self.cube_size, self.cube_size, self.cube_size],
                rgba=rgba,
                material=map_textures[color_name],
            )
            
            self.cubes.append(cube)
            self.cube_colors.append(color_idx)

        # Create sorting bins (colored platforms)
        self.bins = []
        self.bin_positions = []
        
        # Bins are positioned in a row on the left side
        bin_y_start = -0.2
        bin_y_spacing = 0.2
        bin_x = -0.1
        
        for i in range(self.num_bins):
            # Create a flat box object to serve as a colored platform/bin
            color_name, rgba = self.color_categories[i]
            bin_obj = BoxObject(
                name=f"bin{i}",
                size_min=[0.06, 0.06, 0.001],  # Flat platform: 12cm x 12cm x 1cm
                size_max=[0.06, 0.06, 0.001],
                rgba=rgba,  # Match the color category
                obj_type="visual",  # Make it visual only
                joints=None,  # No joints
            )
            self.bins.append(bin_obj)
            
            bin_y = bin_y_start + i * bin_y_spacing
            self.bin_positions.append(np.array([bin_x, bin_y, self.table_offset[2] + 0.005]))

        # Assembly line position (right side, where cubes spawn)
        self.assembly_line_x = 0.0
        self.assembly_line_y_start = -0.25
        self.assembly_line_y_end = 0.25
        
        self.objects = self.cubes + self.bins

        # Create placement initializers for bins
        self.bin_placement_initializer = SequentialCompositeSampler(name="BinSampler")
        
        for i, (bin_obj, bin_pos) in enumerate(zip(self.bins, self.bin_positions)):
            self.bin_placement_initializer.append_sampler(
                UniformRandomSampler(
                    name=f"Bin{i}Sampler",
                    mujoco_objects=bin_obj,
                    x_range=[bin_pos[0], bin_pos[0]],
                    y_range=[bin_pos[1], bin_pos[1]],
                    rotation=0,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.table_offset,
                    z_offset=0.0,  # Place directly on table
                ))

        # Create placement initializers for cubes on assembly line
        self.cube_placement_initializers = []
        
        for i, cube in enumerate(self.cubes):
            sampler = UniformRandomSampler(
                name=f"Cube{i}Sampler",
                mujoco_objects=cube,
                x_range=[self.assembly_line_x, self.assembly_line_x],
                y_range=[self.assembly_line_y_start, self.assembly_line_y_end],
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

        # Randomly reassign cube colors on each reset
        # self.cube_colors = []
        # for i in range(self.num_cubes):
        #     color_idx = np.random.randint(0, self.num_bins)
        #     self.cube_colors.append(color_idx)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for bins
            bin_placements = self.bin_placement_initializer.sample()
            for obj_pos, obj_quat, obj in bin_placements.values():
                body_id = self.sim.model.body_name2id(obj.root_body)
                self.sim.model.body_pos[body_id] = obj_pos
                self.sim.model.body_quat[body_id] = obj_quat

            # Randomly assign positions along assembly line for cubes
            y_positions = np.linspace(self.assembly_line_y_start, self.assembly_line_y_end, self.num_cubes)
            np.random.shuffle(y_positions)
            
            # Sample from the placement initializer for cubes
            for i, (cube, sampler, color_idx) in enumerate(zip(self.cubes, self.cube_placement_initializers, self.cube_colors)):
                # Update cube color
                color_name, rgba = self.color_categories[color_idx]
                # Note: Changing rgba in MuJoCo requires modifying the geom directly
                # This is a limitation - colors are set at model load time
                # For full color changes, you'd need to reload the model
                
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

                @sensor(modality=modality)
                def cube_to_target_bin(obs_cache, cube_idx=i):
                    cube_pos_key = f"cube{cube_idx}_pos"
                    if cube_pos_key in obs_cache:
                        target_bin_pos = self.bin_positions[self.cube_colors[cube_idx]]
                        return target_bin_pos - obs_cache[cube_pos_key]
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
                
                observables[f"cube{i}_to_target_bin"] = Observable(
                    name=f"cube{i}_to_target_bin",
                    sensor=cube_to_target_bin,
                    sampling_rate=self.control_freq,
                )

            # Add bin position observables
            for i in range(self.num_bins):
                @sensor(modality=modality)
                def bin_pos(obs_cache, bin_idx=i):
                    return self.bin_positions[bin_idx]

                observables[f"bin{i}_pos"] = Observable(
                    name=f"bin{i}_pos",
                    sensor=bin_pos,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _check_success(self):
        """
        Check if all cubes are correctly sorted into bins.

        Returns:
            bool: True if all cubes are correctly sorted
        """
        for i in range(self.num_cubes):
            cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
            target_bin_idx = self.cube_colors[i]
            target_bin_pos = self.bin_positions[target_bin_idx]
            
            if not self._check_in_bin(cube_pos, target_bin_pos):
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
            # Find nearest cube that hasn't been sorted yet
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            min_dist = float('inf')
            nearest_cube = None
            
            for i, cube in enumerate(self.cubes):
                cube_pos = self.sim.data.body_xpos[self.cube_body_ids[i]]
                target_bin_pos = self.bin_positions[self.cube_colors[i]]
                
                # Only consider cubes not yet sorted
                if not self._check_in_bin(cube_pos, target_bin_pos):
                    dist = np.linalg.norm(gripper_site_pos - cube_pos)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_cube = cube
            
            if nearest_cube is not None:
                self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=nearest_cube)
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


class PatternReplication(SingleArmEnv):
    """
    This class corresponds to a pattern replication task for a single robot arm.
    A reference pattern of cubes is displayed on one platform, and the agent must
    replicate that exact pattern on a target platform using loose cubes.

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

        num_cubes (int): Number of cubes in the pattern (default: 3)
        
        pattern_type (str): Type of pattern to replicate. Options: "line", "tower", "pyramid", "random2d"
            - "line": Cubes in a horizontal line
            - "tower": Cubes stacked vertically
            - "pyramid": 2-level pyramid structure
            - "random2d": Random 2D arrangement on platform

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
        num_cubes=3,
        deterministic_reset=False,
        pattern_type="tower",
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
        self.pattern_type = pattern_type
        self.cube_size = 0.02
        self.deterministic_reset = deterministic_reset
        
        # Define colors for cubes
        self.cube_colors = [
            [1, 0, 0, 1],      # Red
            [0, 1, 0, 1],      # Green
            [0, 0, 1, 1],      # Blue
            [1, 1, 0, 1],      # Yellow
            [1, 0, 1, 1],      # Magenta
            [0, 1, 1, 1],      # Cyan
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
        
        # Platform positions - both on same x line
        self.line_x = 0.05
        self.reference_platform_pos = np.array([self.line_x, -0.25, self.table_offset[2] + 0.005])
        self.target_platform_pos = np.array([self.line_x, 0.25, self.table_offset[2] + 0.005])
        
        # Tolerance for position matching (2cm)
        self.position_tolerance = 0.02

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
        - Placing cubes on the target platform
        - Matching the reference pattern positions
        - Matching the color arrangement

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0
        
        # Get positions of movable cubes relative to target platform
        movable_positions = []
        for i in range(self.num_cubes):
            cube_pos = self.sim.data.body_xpos[self.movable_cube_body_ids[i]]
            relative_pos = cube_pos - self.target_platform_pos
            movable_positions.append(relative_pos)
        
        if self.reward_shaping:
            # Dense reward components
            for i in range(self.num_cubes):
                cube_pos = self.sim.data.body_xpos[self.movable_cube_body_ids[i]]
                gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
                
                # Check if cube is on target platform
                on_platform = self._check_on_platform(cube_pos, self.target_platform_pos)
                
                # Find the best matching reference position for this cube
                best_match_reward = 0.0
                if on_platform:
                    # Reward for being on platform
                    reward += 0.5
                    
                    # Additional reward for position accuracy
                    target_pos = self.reference_pattern_positions[i]
                    actual_relative_pos = cube_pos - self.target_platform_pos
                    position_error = np.linalg.norm(actual_relative_pos - target_pos)
                    
                    if position_error < self.position_tolerance:
                        # Perfect match
                        reward += 1.5
                    else:
                        # Partial reward based on closeness
                        reward += 0.5 * (1 - np.tanh(5.0 * position_error))
                else:
                    # Small reward for approaching target platform with cube
                    grasping = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.movable_cubes[i])
                    if grasping:
                        dist_to_platform = np.linalg.norm(cube_pos[:2] - self.target_platform_pos[:2])
                        reward += 0.3 * (1 - np.tanh(2.0 * dist_to_platform))
                    else:
                        # Tiny reward for approaching cube
                        dist_to_cube = np.linalg.norm(gripper_site_pos - cube_pos)
                        reward += 0.1 * (1 - np.tanh(10.0 * dist_to_cube))
        else:
            # Sparse reward: count correctly placed cubes
            num_correct = 0
            
            # Check each movable cube against each reference position
            for i in range(self.num_cubes):
                cube_pos = self.sim.data.body_xpos[self.movable_cube_body_ids[i]]
                
                if self._check_on_platform(cube_pos, self.target_platform_pos):
                    # Check if it matches any reference position
                    actual_relative_pos = cube_pos - self.target_platform_pos
                    
                    for ref_pos in self.reference_pattern_positions:
                        position_error = np.linalg.norm(actual_relative_pos - ref_pos)
                        if position_error < self.position_tolerance:
                            num_correct += 1
                            break
            
            reward = float(num_correct)

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
        z_on = cube_pos[2] > platform_pos[2]  # Above platform
        
        return x_on and y_on and z_on

    def _generate_pattern(self):
        """
        Generate a pattern of relative positions based on pattern_type.
        
        Returns:
            list: List of relative positions [x, y, z] for each cube
        """
        patterns = []
        
        if self.pattern_type == "line":
            # Horizontal line of cubes
            spacing = self.cube_size * 2.2
            start_offset = -(self.num_cubes - 1) * spacing / 2
            for i in range(self.num_cubes):
                x_offset = start_offset + i * spacing
                patterns.append(np.array([x_offset, 0.0, self.cube_size + 0.005]))
                np.random.shuffle(patterns)  # Shuffle order to avoid always placing in same order
                
        elif self.pattern_type == "tower":
            # Vertical stack - first cube on platform, rest stacked on top
            for i in range(self.num_cubes):
                # Each cube stacked on top of previous
                z_offset = i * self.cube_size * 2 #self.cube_size + 0.005 +
                patterns.append(np.array([0.0, 0.0, z_offset]))
                np.random.shuffle(patterns)  # Shuffle order to avoid always stacking in same order
                
        elif self.pattern_type == "pyramid":
            # Simple 2-level pyramid (base + top)
            if self.num_cubes >= 2:
                # Base cubes
                base_spacing = self.cube_size * 2.2
                num_base = min(self.num_cubes - 1, 2)
                for i in range(num_base):
                    x_offset = (i - num_base / 2 + 0.5) * base_spacing
                    patterns.append(np.array([x_offset, 0.0, self.cube_size + 0.005]))
                
                # Top cube(s) - stacked on base
                remaining = self.num_cubes - num_base
                for i in range(remaining):
                    patterns.append(np.array([0.0, 0.0, self.cube_size * 3 + 0.005]))
            else:
                # Just one cube in center
                patterns.append(np.array([0.0, 0.0, self.cube_size + 0.005]))
                
        elif self.pattern_type == "random2d":
            # Random 2D arrangement on platform
            max_offset = 0.04  # Stay within platform bounds
            for i in range(self.num_cubes):
                x_offset = np.random.uniform(-max_offset, max_offset)
                y_offset = np.random.uniform(-max_offset, max_offset)
                patterns.append(np.array([x_offset, y_offset, self.cube_size + 0.005]))
        
        return patterns

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

        # Create reference cubes (static bodies that show the pattern)
        self.reference_cubes = []
        for i in range(self.num_cubes):
            rgba = self.cube_colors[i % len(self.cube_colors)]
            # Make reference cubes semi-transparent
            rgba_transparent = rgba.copy()
            rgba_transparent[3] = 0.5
            for key in self.rgba_semantic_colors:
                if rgba == self.rgba_semantic_colors[key]:
                    color = key
                    break
            
            cube = BoxObject(
                name=f"ref_cube{i}",
                size_min=[self.cube_size, self.cube_size, self.cube_size],
                size_max=[self.cube_size, self.cube_size, self.cube_size],
                rgba=rgba_transparent,
                material=map_textures[key],
                density=10000,  # Make them very heavy so they don't move
            )
            self.reference_cubes.append(cube)
        
        # Create movable cubes (to replicate the pattern) - opaque
        self.movable_cubes = []
        self.movable_cube_body_ids = []

        for i in range(self.num_cubes):
            rgba = self.cube_colors[i % len(self.cube_colors)]
            for key in self.rgba_semantic_colors:
                if rgba == self.rgba_semantic_colors[key]:
                    color = key
                    break
            cube = BoxObject(
                name=f"movable_cube{i}",
                size_min=[self.cube_size, self.cube_size, self.cube_size],
                size_max=[self.cube_size, self.cube_size, self.cube_size],
                rgba=rgba,
                material=map_textures[color],
            )
            self.movable_cubes.append(cube)

        # Create platforms
        self.reference_platform = BoxObject(
            name="reference_platform",
            size_min=[0.06, 0.06, 0.001],
            size_max=[0.06, 0.06, 0.001],
            rgba=[0.3, 0.3, 0.3, 0.7],  # Dark gray, semi-transparent
            obj_type="visual",
            joints=None,
        )
        
        self.target_platform = BoxObject(
            name="target_platform",
            size_min=[0.06, 0.06, 0.001],
            size_max=[0.06, 0.06, 0.001],
            rgba=[0.7, 0.7, 0.7, 1.0],  # Light gray, opaque
            obj_type="visual",
            joints=None,
        )
        
        # Spawn area for movable cubes (on same x line, between platforms)
        self.spawn_area_x = self.line_x
        self.spawn_area_y_start = -0.1
        self.spawn_area_y_end = 0.1
        
        self.objects = self.reference_cubes + self.movable_cubes + [self.reference_platform, self.target_platform]

        # Create placement initializers for platforms
        self.platform_placement_initializer = SequentialCompositeSampler(name="PlatformSampler")
        
        self.platform_placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ReferencePlatformSampler",
                mujoco_objects=self.reference_platform,
                x_range=[self.reference_platform_pos[0], self.reference_platform_pos[0]],
                y_range=[self.reference_platform_pos[1], self.reference_platform_pos[1]],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.005,
            ))
        
        self.platform_placement_initializer.append_sampler(
            UniformRandomSampler(
                name="TargetPlatformSampler",
                mujoco_objects=self.target_platform,
                x_range=[self.target_platform_pos[0], self.target_platform_pos[0]],
                y_range=[self.target_platform_pos[1], self.target_platform_pos[1]],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.005,
            ))

        # Create placement initializers for movable cubes in spawn area (on same line)
        self.movable_cube_placement_initializers = []
        for cube in self.movable_cubes:
            sampler = UniformRandomSampler(
                name=f"{cube.name}Sampler",
                mujoco_objects=cube,
                x_range=[self.spawn_area_x, self.spawn_area_x],
                y_range=[self.spawn_area_y_start, self.spawn_area_y_end],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            self.movable_cube_placement_initializers.append(sampler)

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
        self.movable_cube_body_ids = []
        for cube in self.movable_cubes:
            self.movable_cube_body_ids.append(self.sim.model.body_name2id(cube.root_body))

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        # Generate pattern for this episode
        self.reference_pattern_positions = self._generate_pattern()
        self.gripper_body = self.sim.model.body_name2id('gripper0_eef')

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for platforms
            platform_placements = self.platform_placement_initializer.sample()
            for obj_pos, obj_quat, obj in platform_placements.values():
                body_id = self.sim.model.body_name2id(obj.root_body)
                self.sim.model.body_pos[body_id] = obj_pos
                self.sim.model.body_quat[body_id] = obj_quat

            # Place reference cubes in the pattern on reference platform
            # Sort by z-height to place bottom cubes first (like in Hanoi)
            sorted_indices = sorted(range(len(self.reference_pattern_positions)), 
                                  key=lambda i: self.reference_pattern_positions[i][2])


            reference_placements = {}
            print()
            for idx in sorted_indices:
                i = idx
                cube = self.reference_cubes[i]
                rel_pos = self.reference_pattern_positions[i]
                
                # Calculate absolute position on table
                abs_x = self.reference_platform_pos[0] + rel_pos[0]
                abs_y = self.reference_platform_pos[1] + rel_pos[1]
                
                # Check if this cube should be stacked on another
                # Find if there's a cube below this one in the pattern
                ref_obj = None
                should_stack = False
                
                if rel_pos[2] > self.cube_size + 0.01:  # If z is high enough, might be stacked
                    # Look for a cube at the same x,y but lower z
                    for j in sorted_indices:
                        if j == i or j not in [sorted_indices[k] for k in range(sorted_indices.index(i))]:
                            continue
                        other_rel_pos = self.reference_pattern_positions[j]
                        # Check if same horizontal position
                        horiz_dist = np.linalg.norm(rel_pos[:2] - other_rel_pos[:2])
                        z_diff = rel_pos[2] - other_rel_pos[2]
                        
                        # Should be stacked if horizontally aligned and height difference is ~2*cube_size
                        if horiz_dist < 0.01 and abs(z_diff - self.cube_size * 2) < 0.01:
                            # This cube should be on top of cube j
                            # reference_placements is like: {cube_name: (pos, quat, obj)}
                            cube_name = f"ref_cube{j}"
                            if cube_name in reference_placements:
                                ref_obj = reference_placements[cube_name][0]  # Get the position tuple
                                print(ref_obj)
                                should_stack = True
                                break
                
                if should_stack and ref_obj is not None:
                    # Stack on top of another cube
                    sampler = UniformRandomSampler(
                        name=f"RefCube{i}StackedSampler",
                        mujoco_objects=cube,
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.01,
                    )
                    placement = sampler.sample(reference=ref_obj, on_top=True)
                else:
                                    # Create sampler
                    sampler = UniformRandomSampler(
                        name=f"RefCube{i}Sampler",
                        mujoco_objects=cube,
                        x_range=[abs_x, abs_x],
                        y_range=[abs_y, abs_y],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.01,
                    )
                    # Place on platform/table, create a dummy reference position
                    #ref_obj = [abs_x, abs_y, self.table_offset[2] + 0.01]
                    placement = sampler.sample()#reference=ref_obj, on_top=False)
                print(f"Placing reference cube {i} at ({abs_x:.3f}, {abs_y:.3f}), stacking: {should_stack}")
                # Update reference_placements dict with this cube's placement
                # placement dict format: {cube_name: (pos, quat, obj)}
                reference_placements.update(placement)
                
                # Set the position using joint
                for obj_pos, obj_quat, obj in placement.values():
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
            
            # Place movable cubes randomly in spawn area (on the same line)
            # Distribute them evenly along the line
            y_positions = np.linspace(self.spawn_area_y_start, self.spawn_area_y_end, self.num_cubes)
            np.random.shuffle(y_positions)
            
            for i, (cube, sampler) in enumerate(zip(self.movable_cubes, self.movable_cube_placement_initializers)):
                # Update sampler with assigned y position
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

            # Create observables for each movable cube
            for i in range(self.num_cubes):
                
                @sensor(modality=modality)
                def movable_cube_pos(obs_cache, cube_idx=i):
                    return np.array(self.sim.data.body_xpos[self.movable_cube_body_ids[cube_idx]])
                
                def reference_cube_pos(obs_cache, cube_idx=i):
                    # Target is the corresponding position in the pattern
                    target_abs_pos = self.target_platform_pos + self.reference_pattern_positions[cube_idx]
                    return target_abs_pos

                @sensor(modality=modality)
                def movable_cube_quat(obs_cache, cube_idx=i):
                    return convert_quat(np.array(self.sim.data.body_xquat[self.movable_cube_body_ids[cube_idx]]), to="xyzw")

                @sensor(modality=modality)
                def gripper_to_movable_cube(obs_cache, cube_idx=i):
                    cube_pos_key = f"movable_cube{cube_idx}_pos"
                    return (
                        obs_cache[cube_pos_key] - obs_cache[f"{pf}eef_pos"]
                        if cube_pos_key in obs_cache and f"{pf}eef_pos" in obs_cache
                        else np.zeros(3)
                    )

                @sensor(modality=modality)
                def movable_cube_to_target(obs_cache, cube_idx=i):
                    cube_pos_key = f"movable_cube{cube_idx}_pos"
                    if cube_pos_key in obs_cache:
                        # Target is the corresponding position in the pattern
                        target_abs_pos = self.target_platform_pos + self.reference_pattern_positions[cube_idx]
                        return target_abs_pos - obs_cache[cube_pos_key]
                    return np.zeros(3)

                # Add observables
                observables[f"movable_cube{i}_pos"] = Observable(
                    name=f"movable_cube{i}_pos",
                    sensor=movable_cube_pos,
                    sampling_rate=self.control_freq,
                )
                
                observables[f"movable_cube{i}_quat"] = Observable(
                    name=f"movable_cube{i}_quat",
                    sensor=movable_cube_quat,
                    sampling_rate=self.control_freq,
                )
                
                observables[f"gripper_to_movable_cube{i}"] = Observable(
                    name=f"gripper_to_movable_cube{i}",
                    sensor=gripper_to_movable_cube,
                    sampling_rate=self.control_freq,
                )
                
                observables[f"movable_cube{i}_to_target"] = Observable(
                    name=f"movable_cube{i}_to_target",
                    sensor=movable_cube_to_target,
                    sampling_rate=self.control_freq,
                )

            # Add platform position observables
            @sensor(modality=modality)
            def reference_platform_pos_obs(obs_cache):
                return self.reference_platform_pos

            @sensor(modality=modality)
            def target_platform_pos_obs(obs_cache):
                return self.target_platform_pos

            observables["reference_platform_pos"] = Observable(
                name="reference_platform_pos",
                sensor=reference_platform_pos_obs,
                sampling_rate=self.control_freq,
            )
            
            observables["target_platform_pos"] = Observable(
                name="target_platform_pos",
                sensor=target_platform_pos_obs,
                sampling_rate=self.control_freq,
            )

        return observables

    def _check_success(self):
        """
        Check if the pattern has been successfully replicated.

        Returns:
            bool: True if all cubes match the reference pattern
        """
        num_correct = 0
        
        for i in range(self.num_cubes):
            cube_pos = self.sim.data.body_xpos[self.movable_cube_body_ids[i]]
            
            if self._check_on_platform(cube_pos, self.target_platform_pos):
                # Check if it matches the corresponding reference position
                actual_relative_pos = cube_pos - self.target_platform_pos
                target_relative_pos = self.reference_pattern_positions[i]
                
                position_error = np.linalg.norm(actual_relative_pos - target_relative_pos)
                if position_error < self.position_tolerance:
                    num_correct += 1
        
        return num_correct == self.num_cubes

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
            # Find nearest movable cube not correctly placed
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            min_dist = float('inf')
            nearest_cube = None
            
            for i in range(self.num_cubes):
                cube_pos = self.sim.data.body_xpos[self.movable_cube_body_ids[i]]
                
                # Only consider cubes not yet correctly placed
                on_platform = self._check_on_platform(cube_pos, self.target_platform_pos)
                if on_platform:
                    actual_relative_pos = cube_pos - self.target_platform_pos
                    target_relative_pos = self.reference_pattern_positions[i]
                    position_error = np.linalg.norm(actual_relative_pos - target_relative_pos)
                    if position_error < self.position_tolerance:
                        continue  # Skip correctly placed cubes
                
                dist = np.linalg.norm(gripper_site_pos - cube_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cube = self.movable_cubes[i]
            
            if nearest_cube is not None:
                self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=nearest_cube)
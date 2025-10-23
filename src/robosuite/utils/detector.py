import numpy as np
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
from robosuite.utils.mjcf_utils import find_elements
import matplotlib.pyplot as plt

class PickPlaceDetector:
    def __init__(self, env, single_object_mode=True, object_to_use='can'):
        self.env = env
        if single_object_mode:
            self.objects = [object_to_use, 'door']
        else:
            self.objects = ['cereal', 'milk', 'can', 'bread', 'door']
        self.object_id = {'cereal': 'Cereal_main', 'milk': 'Milk_main', 'can': 'Can_main', 'bread': 'Bread_main', 'door': 'Door_main'}
        self.object_areas = ['pick', 'drop']
        self.grippers_areas = ['pick', 'drop', 'activate', 'lightswitch']
        self.grippers = ['gripper']
        self.area_pos = {'pick': env.bin1_pos, 'drop': env.bin2_pos, 'activate': self.env.activate_pos, 'lightswitch': self.env.lightswitch_pos}
        self.area_size = self.env.ray_bins
        self.active_objs = [self.env.obj_to_use]
        self.max_distance = 10 #max distance for the robotic arm in meters

    def at(self, obj, area, return_distance=False):
        if obj in ['cereal', 'milk', 'can', 'bread']:
            obj_pos = self.env.sim.data.body_xpos[self.env.obj_body]
            if area == 'pick':
                dist = np.linalg.norm(obj_pos - self.area_pos['pick'])
            elif area == 'drop':
                dist = np.linalg.norm(obj_pos - self.area_pos['drop'])
            elif area == 'activate':
                dist = np.linalg.norm(obj_pos - self.area_pos['activate'])
            else:
                return None
        elif obj == 'door':
            return None
        else:
            return None

        if return_distance:
            return dist
        else:
            return bool(dist < self.area_size[area])

    def at_gripper(self, gripper, area, return_distance=False):
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
            if area == 'pick':
                dist = np.linalg.norm(gripper_pos - self.area_pos['pick'])
            elif area == 'drop':
                dist = np.linalg.norm(gripper_pos - self.area_pos['drop'])
            elif area == 'activate':
                dist = np.linalg.norm(gripper_pos - self.area_pos['activate'])
            elif area == 'lightswitch':
                dist = np.linalg.norm(gripper_pos - self.area_pos['lightswitch'])
            else:
                raise ValueError('Invalid area.')
        else:
            raise ValueError('Invalid object.')

        if return_distance:
            return dist
        else:
            return bool(dist < self.area_size[area])

    def grasped(self, obj):
        if obj == 'door':
            return None
        active_obj = self.select_object(obj)

        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True

    def picked_up(self, obj, return_distance=False):
        if obj == 'door':
            return None
        active_obj = self.select_object(obj)
        z_target = self.env.bin1_pos[2] + 0.25
        object_z_loc = self.env.sim.data.body_xpos[self.env.obj_body_id[active_obj.name]][2]
        z_dist = z_target - object_z_loc
        if return_distance:
            return z_dist
        else:
            return bool(z_dist < 0.15)

    def dropped_off(self):
        """
        Returns True if the object is in the correct bin, False otherwise.
        """
        gripper_site_pos = self.env.sim.data.site_xpos[self.env.sim.model.body_name2id('gripper0_eef')]
        for i, obj in enumerate(self.env.objects):
            obj_str = obj.name
            obj_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            self.env.objects_in_bins[i] = float((not self.env.not_in_bin(obj_pos, i)) and r_reach < 0.6)

        # returns True if a single object is in the correct bin
        if self.env.single_object_mode in {1, 2}:
            return bool(np.sum(self.env.objects_in_bins) > 0)

        # returns True if all objects are in correct bins
        return bool(np.sum(self.env.objects_in_bins) == len(self.env.objects))

    def open(self, obj, return_distance=False):
        """
        Selects the object tuple (object, object name) from self.objects using one of the strings ["milk", "bread", "cereal", "can"],
        ignoring the caps from the first letter in the self.obj_names.
        """
        if obj == 'door':
            """
            Returns True if the door is open (i.e., unlocked and open), False otherwise.
            """
            # Get the hinge joint element from the XML
            hinge = find_elements(root=self.env.door.worldbody, tags="joint", attribs={"name": self.env.door.hinge_joint}, return_first=True)

            # Check if the hinge joint is unlocked and open
            if self.env.door.lock:
                return 0 if return_distance else False
            else:
                # If the door is not locked, check if the hinge joint is at its minimum position
                # Get the current position of the hinge joint
                qpos = self.env.sim.data.qpos[self.env.sim.model.get_joint_qpos_addr(f"{obj.capitalize()}_hinge")]

                # Get the closed position of the hinge joint
                qpos_min = hinge.get("range").split(" ")[0]

                # Calculate the relative door aperture as a percentage of the range between closed and maximum positions
                qpos_max = hinge.get("range").split(" ")[1]
                relative_aperture = ((float(qpos) - float(qpos_min)) / (float(qpos_max) - float(qpos_min))) * 100
                return relative_aperture / 100 if return_distance else bool(relative_aperture > 10)

        elif obj == 'gripper':
            """
            Returns True if the gripper is open, False otherwise.
            """
            gripper = self.env.robots[0].gripper
            # Print gripper aperture
            left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
            right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
            aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
            #print(f'Gripper aperture: {aperture}')
            return bool(aperture > 0.13)
        return None

    def door_locked(self):
        """
        Returns True if the door is locked, False otherwise.
        """
        if self.env.door.lock and not(self.open('door')):
            return True
        else:
            return False

    def over(self, gripper, obj, return_distance=False):
        """
        Returns True if the gripper is over the object, False otherwise.
        """
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
            obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
            if return_distance:
                return dist_xy
            else:
                return bool(dist_xy < 0.1)#bool(dist_xy < 0.05)
        else:
            return None

    def at_grab_level(self, gripper, obj, return_distance=False):
        """
        Returns True if the gripper is at the same height as the object, False otherwise.
        """
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
            obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_z = np.linalg.norm(gripper_pos[2] - obj_pos[2])
            if return_distance:
                return dist_z
            else:
                return bool(dist_z < 0.005)
        else:
            return None

    def door_collision(self):
        """
        Returns True if the gripper is colliding with the door, False otherwise.
        """
        active_obj = self.select_object('door')
        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms
        o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        g_group = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        
        return self.env.check_contact(g_group, o_geoms)

    def select_object(self, obj_name):
        """
        Selects the object tuple (object, object name) from self.objects using one of the strings ["milk", "bread", "cereal", "can"],
        ignoring the caps from the first letter in the self.obj_names.
        """
        obj_name = obj_name.lower().capitalize()
        for obj, name in zip(self.env.objects, self.env.obj_names):
            if name.startswith(obj_name):
                return obj
        return None
    
    def get_all_objects_pos(self):
        # Return a dict of all object positions
        positions = {}
        for obj in self.objects:
            body_id = self.env.sim.model.body_name2id(self.object_id[obj])
            positions[obj] = np.asarray(self.env.sim.data.body_xpos[body_id])
        # Add eef position
        positions['gripper'] = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
        return positions

    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
        
        groundings = {}

        # Check if each object is in each area
        for obj in self.objects:
            if obj == 'door':
                continue
            for area in self.object_areas:
                at_value = self.at(obj, area, return_distance=return_distance)
                if not(self.env.light_on):
                    at_value = self.max_distance if return_distance else False
                #if return_distance:
                #    at_value = at_value / self.max_distance  # Normalize distance
                if binary_to_float:
                    at_value = float(at_value)
                groundings[f'at({obj},{area})'] = at_value

        # Check if the gripper is in each area and if it's free
        for gripper in self.grippers:
            for area in self.grippers_areas:
                at_gripper_value = self.at_gripper(gripper, area, return_distance=return_distance)
                if not(self.env.light_on):
                    at_gripper_value = self.max_distance if return_distance else False
                #if return_distance:
                #    at_gripper_value = at_gripper_value / self.max_distance  # Normalize distance
                if binary_to_float:
                    at_gripper_value = float(at_gripper_value)
                groundings[f'at_gripper({gripper},{area})'] = at_gripper_value

            # Check if the gripper is grasping each object
            for obj in self.objects:
                if obj == 'door':
                    continue
                grasped_value = self.grasped(obj)
                if not(self.env.light_on):
                    grasped_value = False
                if binary_to_float:
                    grasped_value = float(grasped_value)
                groundings[f'grasped({obj})'] = grasped_value

        # Check if the door is open
        door_open_value = self.open('door', return_distance=return_distance)
        if not(self.env.light_on):
            door_open_value = self.max_distance if return_distance else False
        if binary_to_float:
            door_open_value = float(door_open_value)
        groundings['open(door)'] = door_open_value

        # Check if the door is locked
        door_locked_value = self.door_locked()
        if not(self.env.light_on):
            door_locked_value = True
        if binary_to_float:
            door_locked_value = float(door_locked_value)
        groundings['locked(door)'] = door_locked_value

        # Check if the gripper is open
        gripper_open_value = self.open('gripper')
        if binary_to_float:
            gripper_open_value = float(gripper_open_value)
        groundings['open(gripper)'] = gripper_open_value

        # Check if the gripper is colliding with the door
        door_collision_value = self.door_collision()
        if binary_to_float:
            door_collision_value = float(door_collision_value)
        groundings['door_collision'] = door_collision_value

        # Check if an object has been picked up
        for obj in self.objects:
            if obj == 'door':
                continue
            picked_up_value = self.picked_up(obj, return_distance=return_distance)
            #if return_distance:
            #    picked_up_value = picked_up_value / self.max_distance  # Normalize distance
            if binary_to_float:
                picked_up_value = float(picked_up_value)
            groundings[f'picked_up({obj})'] = picked_up_value

        # Check if an object has been dropped off
        dropped_off_value = self.dropped_off()
        if binary_to_float:
            dropped_off_value = float(dropped_off_value)
        groundings['dropped_off'] = dropped_off_value

        # Check if the gripper is over each object
        for gripper in self.grippers:
            for obj in self.objects:
                if obj == 'door':
                    continue
                over_value = self.over(gripper, obj, return_distance=return_distance)
                if not(self.env.light_on):
                    over_value = self.max_distance if return_distance else False
                #if return_distance:
                #    over_value = over_value / self.max_distance
                if binary_to_float:
                    over_value = float(over_value)
                groundings[f'over({gripper},{obj})'] = over_value

        # Check if the gripper is at the same height as each object
        for gripper in self.grippers:
            for obj in self.objects:
                if obj == 'door':
                    continue
                at_grab_level_value = self.at_grab_level(gripper, obj, return_distance=return_distance)
                #if return_distance:
                #    at_grab_level_value = at_grab_level_value / self.max_distance
                if binary_to_float:
                    at_grab_level_value = float(at_grab_level_value)
                groundings[f'at_grab_level({gripper},{obj})'] = at_grab_level_value
        
        if not(self.env.light_on):
            groundings['light_off'] = 1.0 if binary_to_float else True
        else:
            groundings['light_off'] = 0.0 if binary_to_float else False

        #print(groundings)
        #print()
        return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])
    
    def dict_to_array(self, groundings):
        return np.asarray([v for k, v in sorted(groundings.items())])


class HanoiDetector:
    def __init__(self, env):
        self.env = env
        self.objects = ['cube1', 'cube2', 'cube3']
        self.object_id = {'cube1': 'cube1_main', 'cube2': 'cube2_main', 'cube3': 'cube3_main', 'peg1': 'peg1_main', 'peg2': 'peg2_main', 'peg3': 'peg3_main'}
        self.object_areas = ['peg1', 'peg2', 'peg3']
        self.area_pos = {'peg1': self.env.pegs_xy_center[0], 'peg2': self.env.pegs_xy_center[1], 'peg3': self.env.pegs_xy_center[2]}
        self.grippers_areas = ['pick', 'drop', 'activate', 'lightswitch']
        self.grippers = ['gripper']
        self.area_size = self.env.peg_radius
        self.max_distance = 10 #max distance for the robotic arm in meters

    def at(self, obj, area, return_distance=False):
        obj_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj]]
        dist = np.linalg.norm(obj_pos - self.area_pos[area])
        if return_distance:
            return dist
        else:
            return bool(dist < self.area_size)

    def select_object(self, obj_name):
        """
        Selects the object tuple (object, object name) from self.objects using one of the strings ["milk", "bread", "cereal", "can"],
        ignoring the caps from the first letter in the self.obj_names.
        """
        obj_name = obj_name.lower()
        for obj, name in zip(self.env.objects, self.env.obj_names):
            if name.startswith(obj_name):
                return obj
        return None

    def grasped(self, obj):
        active_obj = self.select_object(obj)

        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True

    def over(self, gripper, obj, return_distance=False):
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
            if obj in self.object_areas:
                obj_pos = self.area_pos[obj]
            else:
                obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
            if return_distance:
                return dist_xy
            else:
                return bool(dist_xy < 0.004)#bool(dist_xy < 0.02)#bool(dist_xy < 0.004)#return bool(dist_xy < 0.004)
    
    def at_grab_level(self, gripper, obj, return_distance=False):
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
            obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_z = np.linalg.norm(gripper_pos[2] - obj_pos[2])
            if return_distance:
                return dist_z
            else:
                return bool(dist_z < 0.005)
    
    def on(self, obj1, obj2):
        obj1_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj1]]
        if obj2 in self.object_areas:
            obj2_pos = self.area_pos[obj2]
        else:
            obj2_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj2]]
        dist_xyz = np.linalg.norm(obj1_pos - obj2_pos)
        dist_xy = np.linalg.norm(obj1_pos[:-1] - obj2_pos[:-1])
        dist_z = np.linalg.norm(obj1_pos[2] - obj2_pos[2])
        return bool(dist_xy < 0.03 and obj1_pos[2] > obj2_pos[2]+0.001 and dist_z < 0.055)
    
    def clear(self, obj):
        for other_obj in self.objects:
            if other_obj != obj:
                if self.on(other_obj, obj):
                    return False
        return True
    
    def open(self, gripper, return_distance=False):
        if gripper == 'gripper':
            """
            Returns True if the gripper is open, False otherwise.
            """
            gripper = self.env.robots[0].gripper
            # Print gripper aperture
            left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
            right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
            aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
            #print(f'Gripper aperture: {aperture}')
            if return_distance:
                return aperture
            return bool(aperture > 0.13)
        else:
            return None
    
    def picked_up(self, obj, return_distance=False):
        active_obj = self.select_object(obj)
        z_target = self.env.table_offset[2] + 0.25
        object_z_loc = self.env.sim.data.body_xpos[self.env.obj_body_id[active_obj.name]][2]
        z_dist = z_target - object_z_loc
        if return_distance:
            return z_dist
        else:
            return bool(z_dist < 0.15)

    def get_all_objects_pos(self):
        # Return a dict of all object positions
        positions = {}
        for obj in self.objects:
            body_id = self.env.sim.model.body_name2id(self.object_id[obj])
            positions[obj] = np.asarray(self.env.sim.data.body_xpos[body_id])
        # Add eef position
        positions['gripper'] = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
        return positions

    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
            
            groundings = {}
    
            # Check if the gripper is grasping each object
            for obj in self.objects:
                grasped_value = self.grasped(obj)
                if binary_to_float:
                    grasped_value = float(grasped_value)
                groundings[f'grasped({obj})'] = grasped_value

            # Check if the gripper is over each object
            for gripper in ['gripper']:
                for obj in self.objects+self.object_areas:
                    over_value = self.over(gripper, obj, return_distance=return_distance)
                    if return_distance:
                        over_value = over_value / self.max_distance
                    if binary_to_float:
                        over_value = float(over_value)
                    groundings[f'over({gripper},{obj})'] = over_value

            # Check if the gripper is at the same height as each object
            for gripper in ['gripper']:
                for obj in self.objects:
                    at_grab_level_value = self.at_grab_level(gripper, obj, return_distance=return_distance)
                    if return_distance:
                        at_grab_level_value = at_grab_level_value / self.max_distance
                    if binary_to_float:
                        at_grab_level_value = float(at_grab_level_value)
                    groundings[f'at_grab_level({gripper},{obj})'] = at_grab_level_value

            # Check if each object is on another object
            for obj1 in self.objects:
                for obj2 in self.objects+self.object_areas:
                    if obj1 != obj2:
                        on_value = self.on(obj1, obj2)
                        if binary_to_float:
                            on_value = float(on_value)
                        groundings[f'on({obj1},{obj2})'] = on_value

            # Check if each object is clear
            for obj in self.objects+self.object_areas:
                clear_value = self.clear(obj)
                if binary_to_float:
                    clear_value = float(clear_value)
                groundings[f'clear({obj})'] = clear_value
            
            # Check if the gripper is open
            gripper_open_value = self.open('gripper', return_distance=return_distance)
            if binary_to_float:
                gripper_open_value = float(gripper_open_value)
            groundings['open_gripper(gripper)'] = gripper_open_value

            # Check if an object has been picked up
            for obj in self.objects:
                picked_up_value = self.picked_up(obj, return_distance=return_distance)
                if return_distance:
                    picked_up_value = picked_up_value / self.max_distance
                if binary_to_float:
                    picked_up_value = float(picked_up_value)
                groundings[f'picked_up({obj})'] = picked_up_value

            return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])
    
    def dict_to_array(self, groundings):
        return np.asarray([v for k, v in sorted(groundings.items())])


class NutAssemblyDetector:
    def __init__(self, env):
        self.env = env
        self.objects = ['roundnut', 'squarenut']
        self.object_id = {'roundnut': 'RoundNut_main', 'squarenut': 'SquareNut_main', 'squarepeg': 'peg1', 'roundpeg': 'peg2', 'gripper': 'gripper0_eef'}
        self.grippers = ['gripper']
        self.max_distance = 10 #max distance for the robotic arm in meters

    def select_object(self, obj_name):
        """
        Selects the object tuple (object, object name) from self.objects using one of the strings ["milk", "bread", "cereal", "can"],
        ignoring the caps from the first letter in the self.obj_names.
        """
        obj_name = obj_name.lower()
        for obj in self.env.nuts:
            if obj.name.lower().startswith(obj_name):
                return obj
        return None

    def grasped(self, obj):
        active_obj = self.select_object(obj)

        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True

    def over(self, gripper, obj, return_distance=False):
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
            if "squarepeg" in obj:
                obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.peg1_body_id]) + np.array([-0.02, 0.03, 0])
            else:
                obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body]) + np.array([0, 0.05, 0])
            dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
            if return_distance:
                return dist_xy
            else:
                if "roundpeg" in obj:
                    return bool(dist_xy < 0.03)
                elif "squarepeg" in obj:
                    return bool(dist_xy < 0.01)
                return bool(dist_xy < 0.005)#return bool(dist_xy < 0.004)
    
    def at_grab_level(self, gripper, obj, return_distance=False):
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
            obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_z = np.linalg.norm(gripper_pos[2] - obj_pos[2])
            if return_distance:
                return dist_z
            else:
                return bool(dist_z < 0.005)
    
    def open(self, gripper, return_distance=False):
        if gripper == 'gripper':
            """
            Returns True if the gripper is open, False otherwise.
            """
            left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
            right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
            aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
            if return_distance:
                return aperture
            return bool(aperture > 0.13)
        else:
            return None
    
    def picked_up(self, obj, return_distance=False):
        active_obj = self.select_object(obj)
        z_target = self.env.table_offset[2] + 0.45
        object_z_loc = self.env.sim.data.body_xpos[self.env.obj_body_id[active_obj.name]][2]
        z_dist = z_target - object_z_loc
        if return_distance:
            return z_dist
        else:
            return bool(z_dist < 0.15)

    def on_peg(self, obj_pos, peg_id):

        if peg_id == 0:
            peg_pos = np.array(self.env.sim.data.body_xpos[self.env.peg1_body_id])
        else:
            peg_pos = np.array(self.env.sim.data.body_xpos[self.env.peg2_body_id])
        res = False
        if (
            abs(obj_pos[0] - peg_pos[0]) < 0.03
            and abs(obj_pos[1] - peg_pos[1]) < 0.03
            and obj_pos[2] < self.env.table_offset[2] + 0.16
        ):
            res = True
        return res

    def get_all_objects_pos(self):
        # Return a dict of all object positions
        positions = {}
        for obj in self.objects:
            body_id = self.env.sim.model.body_name2id(self.object_id[obj])
            positions[obj] = np.asarray(self.env.sim.data.body_xpos[body_id])
        # Add eef position
        positions['gripper'] = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
        return positions

    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
            
            groundings = {}
    
            # Check if the gripper is grasping each object
            for obj in self.objects:
                grasped_value = self.grasped(obj)
                if binary_to_float:
                    grasped_value = float(grasped_value)
                groundings[f'grasped({obj})'] = grasped_value

            # Check if the gripper is over each object
            for gripper in ['gripper']:
                for obj in self.objects:
                    over_value = self.over(gripper, obj, return_distance=return_distance)
                    if return_distance:
                        over_value = over_value / self.max_distance
                    if binary_to_float:
                        over_value = float(over_value)
                    groundings[f'over({gripper},{obj})'] = over_value
                for peg in ['roundpeg', 'squarepeg']:
                    over_value = self.over(gripper, peg, return_distance=return_distance)
                    if return_distance:
                        over_value = over_value / self.max_distance
                    if binary_to_float:
                        over_value = float(over_value)
                    groundings[f'over({gripper},{peg})'] = over_value

            # Check if the gripper is at the same height as each object
            for gripper in ['gripper']:
                for obj in self.objects:
                    at_grab_level_value = self.at_grab_level(gripper, obj, return_distance=return_distance)
                    if return_distance:
                        at_grab_level_value = at_grab_level_value / self.max_distance
                    if binary_to_float:
                        at_grab_level_value = float(at_grab_level_value)
                    groundings[f'at_grab_level({gripper},{obj})'] = at_grab_level_value

            # Check if each object is on another object
            for obj in self.objects:
                for i, peg in enumerate(['squarepeg', 'roundpeg']):
                    obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
                    obj_pos = self.env.sim.data.body_xpos[obj_body]
                    on_value = self.on_peg(obj_pos, i)
                    if binary_to_float:
                        on_value = float(on_value)
                    groundings[f'on({obj},{peg})'] = on_value

            # # Check if each object is on another object
            # for obj in self.objects:
            #     for i, peg in enumerate(['squarepeg', 'roundpeg']):
            #         obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
            #         obj_pos = self.env.sim.data.body_xpos[obj_body]
            #         on_value = self.above(obj_pos, i)
            #         if binary_to_float:
            #             on_value = float(on_value)
            #         groundings[f'above({obj},{peg})'] = on_value

            # Check if the gripper is open
            gripper_open_value = self.open('gripper', return_distance=return_distance)
            if binary_to_float:
                gripper_open_value = float(gripper_open_value)
            groundings['open_gripper(gripper)'] = gripper_open_value

            # Check if an object has been picked up
            for obj in self.objects:
                picked_up_value = self.picked_up(obj, return_distance=return_distance)
                if return_distance:
                    picked_up_value = picked_up_value / self.max_distance
                if binary_to_float:
                    picked_up_value = float(picked_up_value)
                groundings[f'picked_up({obj})'] = picked_up_value

            return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])
    
    def dict_to_array(self, groundings):
        return np.asarray([v for k, v in sorted(groundings.items())])


class KitchenDetector:
    def __init__(self, env):
        # Available "body" names = ('world', 'table', 'Stove1_main', 'Stove1_base', 'Stove1_burner', 'Button1_main', 
        # 'Button1_base', 'Button1_switch', 'ServingRegionRed_main', 'ServingRegionRed_base', 'robot0_base', 'robot0_shoulder_link', 
        # 'robot0_HalfArm1_Link', 'robot0_HalfArm2_Link', 'robot0_forearm_link', 'robot0_SphericalWrist1_Link', 
        # 'robot0_SphericalWrist2_Link', 'robot0_Bracelet_Link', 'robot0_right_hand', 'gripper0_robotiq_85_adapter_link', 
        # 'gripper0_eef', 'gripper0_left_outer_knuckle', 'gripper0_left_inner_finger', 'gripper0_left_inner_knuckle', 
        # 'gripper0_right_outer_knuckle', 'gripper0_right_inner_finger', 'gripper0_right_inner_knuckle', 'mount0_base', 
        # 'mount0_controller_box', 'mount0_pedestal_feet', 'mount0_torso', 'mount0_pedestal', 'cube_bread_main', 'PotObject_root').
        self.env = env
        self.objects = ['stove', 'button', 'serving', 'pot', 'bread']
        self.object_id = {'stove': 'Stove1_main', 'button': 'Button1_switch', 'serving': 'ServingRegionRed_main', 'pot': 'PotObject_root', 'bread': 'cube_bread_main'}
        self.grippers = ['gripper']
        #self.area_pos = {'pot': self.env.sim.data.body_xpos[self.env.sim.model.body_name2id()]}
        self.max_distance = 10 #max distance for the robotic arm in meters

    def select_object(self, obj_name):
        """
        Selects the object tuple (object, object name) from self.objects using one of the strings ["milk", "bread", "cereal", "can"],
        ignoring the caps from the first letter in the self.obj_names.
        """
        obj_name = obj_name.lower()
        for obj in self.env.objects + [self.env.button_object_1]:
            if obj_name in obj.name.lower():
                return obj
        print(f'Object {obj_name} not found.')
        return None

    def grasped(self, obj):
        active_obj = self.select_object(obj)

        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True

    def over(self, gripper, obj, return_distance=False):
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
            if obj == 'pot_handle':
                obj_body = self.env.sim.model.body_name2id(self.object_id["pot"])
                obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])+np.array([0, -0.09, 0])
            else:
                obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
                obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
            dist_x = np.linalg.norm(gripper_pos[0] - obj_pos[0])
            if return_distance:
                return dist_xy
            elif "pot" in obj:
                return bool(dist_xy < 0.05)#bool(dist_xy < 0.02)#bool(dist_xy < 0.005)# and bool(dist_x < 0.02)
            elif "stove" in obj:
                return bool(dist_xy < 0.08)#bool(dist_xy < 0.05)#return bool(dist_xy < 0.004)
            elif "serving" in obj:
                return bool(dist_xy < 0.01)
            else:
                return bool(dist_xy < 0.004)
    
    def at_grab_level(self, gripper, obj, return_distance=False):
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
            obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_z = np.linalg.norm(gripper_pos[2] - obj_pos[2])
            if return_distance:
                return dist_z
            elif "pot" in obj:
                return bool(dist_z < 0.04)
            elif "button" in obj:
                return bool(dist_z < 0.09)
            else:
                return bool(dist_z < 0.005)
    
    def open(self, gripper, return_distance=False):
        if gripper == 'gripper':
            """
            Returns True if the gripper is open, False otherwise.
            """
            left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
            right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
            aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
            if return_distance:
                return aperture
            return bool(aperture > 0.13)
        else:
            return None
    
    def picked_up(self, obj, return_distance=False):
        active_obj = self.env.sim.model.body_name2id(self.object_id[obj])
        z_target = self.env.table_offset[2] + 0.25#0.25
        object_z_loc = self.env.sim.data.body_xpos[active_obj][2]
        z_dist = abs(z_target - object_z_loc)
        if return_distance:
            return z_dist
        else:
            return bool(z_dist < 0.05)
    
    def on(self, obj1, obj2):
        obj2_str = obj2.lower()
        obj1 = self.env.sim.model.body_name2id(self.object_id[obj1])
        obj2 = self.env.sim.model.body_name2id(self.object_id[obj2])
        obj1_pos = self.env.sim.data.body_xpos[obj1]
        obj2_pos = self.env.sim.data.body_xpos[obj2]
        dist_xy = np.linalg.norm(obj1_pos[:-1] - obj2_pos[:-1])
        dist_z = np.linalg.norm(obj1_pos[2] - obj2_pos[2])
        if "pot" in obj2_str:
            return bool(dist_xy < 0.04 and obj1_pos[2] > obj2_pos[2]+0.00001 and dist_z < 0.15)
        else:
            #return bool(dist_xy < 0.07 and obj1_pos[2] > obj2_pos[2]+0.00001 and dist_z < 0.055)
            #return bool(dist_xy < 0.055 and obj1_pos[2] > obj2_pos[2]+0.00001 and dist_z < 0.055)
            #print(f"dist_xy: {dist_xy}, dist_z: {dist_z}")
            return bool(dist_xy < 0.07 and dist_z < 0.055)

    def button_on(self):
        """
        Returns True if the button is on, False otherwise.
        """
        return bool(self.env.buttons_on[1])

    def get_all_objects_pos(self):
        # Return a dict of all object positions
        positions = {}
        for obj in self.objects:
            body_id = self.env.sim.model.body_name2id(self.object_id[obj])
            positions[obj] = np.asarray(self.env.sim.data.body_xpos[body_id])
        # Add eef position
        positions['gripper'] = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
        return positions

    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
            
            groundings = {}
    
            # Check if the gripper is grasping each object
            for obj in self.objects:
                grasped_value = self.grasped(obj)
                if binary_to_float:
                    grasped_value = float(grasped_value)
                groundings[f'grasped({obj})'] = grasped_value

            # Check if the gripper is over each object
            for gripper in ['gripper']:
                for obj in self.objects + ['pot_handle']:
                    over_value = self.over(gripper, obj, return_distance=return_distance)
                    if return_distance:
                        over_value = over_value / self.max_distance
                    if binary_to_float:
                        over_value = float(over_value)
                    groundings[f'over({gripper},{obj})'] = over_value

            # Check if the gripper is at the same height as each object
            for gripper in ['gripper']:
                for obj in self.objects:
                    at_grab_level_value = self.at_grab_level(gripper, obj, return_distance=return_distance)
                    if return_distance:
                        at_grab_level_value = at_grab_level_value / self.max_distance
                    if binary_to_float:
                        at_grab_level_value = float(at_grab_level_value)
                    groundings[f'at_grab_level({gripper},{obj})'] = at_grab_level_value

            # Check if each object is on another object
            for obj in self.objects:
                for obj2 in self.objects:
                    if obj != obj2:
                        on_value = self.on(obj, obj2)
                        if binary_to_float:
                            on_value = float(on_value)
                        groundings[f'on({obj},{obj2})'] = on_value

            # Check if the gripper is open
            gripper_open_value = self.open('gripper', return_distance=return_distance)
            if binary_to_float:
                gripper_open_value = float(gripper_open_value)
            groundings['open_gripper(gripper)'] = gripper_open_value

            # Check if an object has been picked up
            for obj in self.objects:
                picked_up_value = self.picked_up(obj, return_distance=return_distance)
                if return_distance:
                    picked_up_value = picked_up_value / self.max_distance
                if binary_to_float:
                    picked_up_value = float(picked_up_value)
                groundings[f'picked_up({obj})'] = picked_up_value
            
            # Check if the button is on
            button_on_value = self.button_on()
            if binary_to_float:
                button_on_value = float(button_on_value)
            for button in ['button']:
                groundings[f'stove_on()'] = button_on_value

            return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])
    
    def dict_to_array(self, groundings):
        return np.asarray([v for k, v in sorted(groundings.items())])


class CubeSortingDetector:
    """
    Detector for CubeSorting environment.
    Task: Sort cubes by size to appropriate platforms (small->platform1, large->platform2).
    """
    def __init__(self, env):
        self.env = env
        self.num_cubes = env.num_cubes
        self.objects = [f'cube{i}' for i in range(self.num_cubes)]
        self.platforms = ['platform1', 'platform2']
        self.grippers = ['gripper']
        self.max_distance = 10  # max distance for the robotic arm in meters
        self.object_id = {f'cube{i}': env.cubes[i].name+"_main" for i in range(self.num_cubes)}
        self.object_id.update({'platform1': 'platform1_main', 'platform2': 'platform2_main'})

    def select_object(self, obj_name):
        """Select cube object by name."""
        for i, cube in enumerate(self.env.cubes):
            if f'cube{i}' == obj_name:
                return cube
        return None

    def grasped(self, obj):
        """Check if object is grasped by gripper."""
        cube_idx = int(obj.replace('cube', ''))
        active_obj = self.env.cubes[cube_idx]
        
        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True

    def over(self, gripper, obj, return_distance=False):
        """Check if gripper is over object or platform."""
        if gripper != 'gripper':
            return None
            
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
        
        if obj in self.platforms:
            # Get platform position
            if obj == 'platform1':
                obj_pos = self.env.platform1_pos
            else:
                obj_pos = self.env.platform2_pos
        else:
            # Get cube position
            cube_idx = int(obj.replace('cube', ''))
            obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]])
        
        dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
        
        if return_distance:
            return dist_xy
        else:
            return bool(dist_xy < 0.005)

    def at_grab_level(self, gripper, obj, return_distance=False):
        """Check if gripper is at same height as object."""
        if gripper != 'gripper' or obj in self.platforms:
            return None
            
        cube_idx = int(obj.replace('cube', ''))
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
        obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]])
        dist_z = abs(gripper_pos[2] - obj_pos[2])
        
        if return_distance:
            return dist_z
        else:
            return bool(dist_z < 0.02)

    def on_platform(self, obj, platform):
        """Check if cube is on specified platform."""
        cube_idx = int(obj.replace('cube', ''))
        cube_pos = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]]
        
        if platform == 'platform1':
            platform_pos = self.env.platform1_pos
        else:
            platform_pos = self.env.platform2_pos
        
        return self.env._check_on_platform(cube_pos, platform_pos)

    def correct_platform(self, obj):
        """Check if cube is on the correct platform based on its size."""
        cube_idx = int(obj.replace('cube', ''))
        size = self.env.cube_sizes[cube_idx]
        cube_pos = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]]
        
        if size == "small":
            return self.env._check_on_platform(cube_pos, self.env.platform1_pos)
        else:  # large
            return self.env._check_on_platform(cube_pos, self.env.platform2_pos)

    def open_gripper(self, return_distance=False):
        """Check if gripper is open."""
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        if return_distance:
            return aperture
        return bool(aperture > 0.13)

    def picked_up(self, obj, return_distance=False):
        """Check if cube has been picked up."""
        cube_idx = int(obj.replace('cube', ''))
        z_target = self.env.table_offset[2] + 0.25
        object_z_loc = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]][2]
        z_dist = z_target - object_z_loc
        
        if return_distance:
            return z_dist
        else:
            return bool(z_dist < 0.1)
        
    def grasped(self, obj):
        """Check if object is grasped by gripper."""
        cube_idx = int(obj.replace('cube', ''))
        active_obj = self.env.cubes[cube_idx]
        
        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True

    def on(self, obj1, obj2):
        """Check if obj1 is on obj2."""
        if obj2 not in self.platforms:
            for i, other_cube in enumerate(self.objects):
                if obj2 == other_cube:
                    other_body = self.env.cube_body_ids[i]
                    break
            else:
                return False
            cube_idx = int(obj1.replace('cube', ''))
            obj_body = self.env.cube_body_ids[cube_idx]
            obj_pos = self.env.sim.data.body_xpos[obj_body]
            other_pos = self.env.sim.data.body_xpos[other_body]
            dist_xy = np.linalg.norm(obj_pos[:-1] - other_pos[:-1])
            dist_z = other_pos[2] - obj_pos[2]
            if dist_xy < 0.05 and dist_z > 0 and dist_z < 0.05:
                return True
            return False
        else:
            return self.on_platform(obj1, obj2)

    def clear(self, obj):
        """Check if no other cube is on top of the specified cube."""
        cube_idx = int(obj.replace('cube', ''))
        obj_body = self.env.cube_body_ids[cube_idx]
        obj_pos = self.env.sim.data.body_xpos[obj_body]
        for i, other_cube in enumerate(self.objects):
            if i != cube_idx:
                other_body = self.env.cube_body_ids[i]
                other_pos = self.env.sim.data.body_xpos[other_body]
                dist_xy = np.linalg.norm(obj_pos[:-1] - other_pos[:-1])
                dist_z = other_pos[2] - obj_pos[2]
                if dist_xy < 0.05 and dist_z > 0 and dist_z < 0.05:
                    return False
        return True

    def small(self, obj1):
        """Check if obj1 is smaller than obj2."""
        cube_idx = int(obj1.replace('cube', ''))
        size = self.env.cube_sizes[cube_idx]
        return size == "small"

    def get_all_objects_pos(self):
        # Return a dict of all object positions
        positions = {}
        for obj in self.objects:
            body_id = self.env.sim.model.body_name2id(self.object_id[obj])
            positions[obj] = np.asarray(self.env.sim.data.body_xpos[body_id])
        # Add eef position
        positions['gripper'] = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
        return positions

    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
        """Get all predicates for the environment state."""
        groundings = {}

        # Small predicates
        for obj in self.objects:
            small_value = self.small(obj)
            if binary_to_float:
                small_value = float(small_value)
            groundings[f'small({obj})'] = small_value

        # Grasped predicates
        for obj in self.objects:
            grasped_value = self.grasped(obj)
            if binary_to_float:
                grasped_value = float(grasped_value)
            groundings[f'grasped({obj})'] = grasped_value

        # Over predicates (gripper over cubes and platforms)
        for obj in self.objects + self.platforms:
            over_value = self.over('gripper', obj, return_distance=return_distance)
            if return_distance:
                over_value = over_value / self.max_distance
            if binary_to_float:
                over_value = float(over_value)
            groundings[f'over(gripper,{obj})'] = over_value

        # At grab level predicates
        for obj in self.objects:
            at_grab_level_value = self.at_grab_level('gripper', obj, return_distance=return_distance)
            if return_distance and at_grab_level_value is not None:
                at_grab_level_value = at_grab_level_value / self.max_distance
            if binary_to_float:
                at_grab_level_value = float(at_grab_level_value)
            groundings[f'at_grab_level(gripper,{obj})'] = at_grab_level_value

        # On platform predicates
        for obj in self.objects:
            for platform in self.platforms:
                on_platform_value = self.on_platform(obj, platform)
                if binary_to_float:
                    on_platform_value = float(on_platform_value)
                groundings[f'on({obj},{platform})'] = on_platform_value

        # Grasped predicates
        for obj in self.objects:
            grasped_value = self.grasped(obj)
            if binary_to_float:
                grasped_value = float(grasped_value)
            groundings[f'grasped({obj})'] = grasped_value

        # Clear predicates
        for obj in self.objects:
            clear_value = self.clear(obj)
            if binary_to_float:
                clear_value = float(clear_value)
            groundings[f'clear({obj})'] = clear_value
        # Add platform clear predicates
        for platform in self.platforms:
            groundings[f'clear({platform})'] = True
        
        # On predicates
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i != j:
                    on_value = self.on(obj1, obj2)
                    if binary_to_float:
                        on_value = float(on_value)
                    groundings[f'on({obj1},{obj2})'] = on_value

        # Correct platform predicates
        for obj in self.objects:
            correct_value = self.correct_platform(obj)
            if binary_to_float:
                correct_value = float(correct_value)
            groundings[f'correct_platform({obj})'] = correct_value

        # Gripper open
        gripper_open_value = self.open_gripper(return_distance=return_distance)
        if binary_to_float:
            gripper_open_value = float(gripper_open_value)
        groundings['open_gripper(gripper)'] = gripper_open_value

        # Picked up predicates
        for obj in self.objects:
            picked_up_value = self.picked_up(obj, return_distance=return_distance)
            if return_distance:
                picked_up_value = picked_up_value / self.max_distance
            if binary_to_float:
                picked_up_value = float(picked_up_value)
            groundings[f'picked_up({obj})'] = picked_up_value

        return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])

    def dict_to_array(self, groundings):
        return np.asarray([v for k, v in sorted(groundings.items())])


class AssemblyLineSortingDetector:
    """
    Detector for AssemblyLineSorting environment.
    Task: Sort cubes from assembly line to color-matched bins.
    """
    def __init__(self, env):
        self.env = env
        self.num_cubes = env.num_cubes
        self.num_bins = env.num_bins
        self.objects = [f'cube{i}' for i in range(self.num_cubes)]
        self.bins = [f'bin{i}' for i in range(self.num_bins)]
        self.grippers = ['gripper']
        self.object_id = {f'cube{i}': env.cubes[i].name+"_main" for i in range(self.num_cubes)}
        self.object_id.update({f'bin{i}': env.bins[i].name+"_main" for i in range(self.num_bins)})
        self.max_distance = 10

    def select_object(self, obj_name):
        """Select cube object by name."""
        for i, cube in enumerate(self.env.cubes):
            if f'cube{i}' == obj_name:
                return cube
        return None

    def grasped(self, obj):
        """Check if object is grasped by gripper."""
        cube_idx = int(obj.replace('cube', ''))
        active_obj = self.env.cubes[cube_idx]
        
        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True
    
    def on(self, obj1, obj2):
        """Check if obj1 is on obj2."""
        if obj2 not in self.bins:
            for i, other_cube in enumerate(self.objects):
                if obj2 == other_cube:
                    other_body = self.env.cube_body_ids[i]
                    break
            else:
                return False
            cube_idx = int(obj1.replace('cube', ''))
            obj_body = self.env.cube_body_ids[cube_idx]
            obj_pos = self.env.sim.data.body_xpos[obj_body]
            other_pos = self.env.sim.data.body_xpos[other_body]
            dist_xy = np.linalg.norm(obj_pos[:-1] - other_pos[:-1])
            dist_z = other_pos[2] - obj_pos[2]
            if dist_xy < 0.05 and dist_z > 0 and dist_z < 0.05:
                return True
            return False
        else:
            cube_idx = int(obj1.replace('cube', ''))
            cube_pos = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]]
            bin_idx = int(obj2.replace('bin', ''))
            bin_pos = self.env.bin_positions[bin_idx]
            return self.env._check_in_bin(cube_pos, bin_pos)
        
    def clear(self, obj):
        """Check if no other cube is on top of the specified cube."""
        cube_idx = int(obj.replace('cube', ''))
        obj_body = self.env.cube_body_ids[cube_idx]
        obj_pos = self.env.sim.data.body_xpos[obj_body]
        for i, other_cube in enumerate(self.objects):
            if i != cube_idx:
                other_body = self.env.cube_body_ids[i]
                other_pos = self.env.sim.data.body_xpos[other_body]
                dist_xy = np.linalg.norm(obj_pos[:-1] - other_pos[:-1])
                dist_z = other_pos[2] - obj_pos[2]
                if dist_xy < 0.05 and dist_z > 0 and dist_z < 0.05:
                    return False
        return True


    def over(self, gripper, obj, return_distance=False):
        """Check if gripper is over object or bin."""
        if gripper != 'gripper':
            return None
            
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
        
        if obj in self.bins:
            # Get bin position
            bin_idx = int(obj.replace('bin', ''))
            obj_pos = self.env.bin_positions[bin_idx]
        else:
            # Get cube position
            cube_idx = int(obj.replace('cube', ''))
            obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]])
        
        dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
        
        if return_distance:
            return dist_xy
        else:
            return bool(dist_xy < 0.005)

    def at_grab_level(self, gripper, obj, return_distance=False):
        """Check if gripper is at same height as object."""
        if gripper != 'gripper' or obj in self.bins:
            return None
            
        cube_idx = int(obj.replace('cube', ''))
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
        obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]])
        dist_z = abs(gripper_pos[2] - obj_pos[2])
        
        if return_distance:
            return dist_z
        else:
            return bool(dist_z < 0.02)

    def in_bin(self, obj, bin_name):
        """Check if cube is in specified bin."""
        cube_idx = int(obj.replace('cube', ''))
        bin_idx = int(bin_name.replace('bin', ''))
        
        cube_pos = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]]
        bin_pos = self.env.bin_positions[bin_idx]
        
        return self.env._check_in_bin(cube_pos, bin_pos)

    def correct_bin(self, obj):
        """Check if cube is in the correct bin based on its color."""
        cube_idx = int(obj.replace('cube', ''))
        target_bin_idx = self.env.cube_colors[cube_idx]
        cube_pos = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]]
        target_bin_pos = self.env.bin_positions[target_bin_idx]
        
        return self.env._check_in_bin(cube_pos, target_bin_pos)

    def open_gripper(self, return_distance=False):
        """Check if gripper is open."""
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        if return_distance:
            return aperture
        return bool(aperture > 0.13)

    def picked_up(self, obj, return_distance=False):
        """Check if cube has been picked up."""
        cube_idx = int(obj.replace('cube', ''))
        z_target = self.env.table_offset[2] + 0.25
        object_z_loc = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]][2]
        z_dist = z_target - object_z_loc
        
        if return_distance:
            return z_dist
        else:
            return bool(z_dist < 0.1)

    def smaller(self, obj1, obj2):
        """Check if obj1 is smaller than obj2."""
        cube_idx1 = int(obj1.replace('cube', ''))
        cube_idx2 = int(obj2.replace('cube', ''))
        size1 = self.env.cube_sizes[cube_idx1]
        size2 = self.env.cube_sizes[cube_idx2]
        
        return size1 < size2
    
    def type_match(self, obj, bin_name):
        """
        Check if the cube's color type matches the bin's color type.
        
        Args:
            obj: cube name (e.g., 'cube0', 'cube1', etc.)
            bin_name: bin name (e.g., 'bin0', 'bin1', 'bin2')
            
        Returns:
            bool: True if cube's color matches bin's color category
        """
        # Extract cube index
        cube_idx = int(obj.replace('cube', ''))
        
        # Extract bin index
        bin_idx = int(bin_name.replace('bin', ''))
        
        # Get the color category index for this cube
        cube_color_idx = self.env.cube_colors[cube_idx]
        #print(f"Cube {obj} color index: {cube_color_idx}, Bin {bin_name} index: {bin_idx}")
        
        # Check if cube's color matches the bin's color category
        # Each bin corresponds to a color category (bin0=red, bin1=green, bin2=blue)
        return cube_color_idx == bin_idx


    def get_all_objects_pos(self):
        # Return a dict of all object positions
        positions = {}
        for obj in self.objects:
            body_id = self.env.sim.model.body_name2id(self.object_id[obj])
            positions[obj] = np.asarray(self.env.sim.data.body_xpos[body_id])
        # Add eef position
        positions['gripper'] = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
        return positions

    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
        """Get all predicates for the environment state."""
        groundings = {}

        # Type match predicates
        for obj in self.objects:
            for bin_name in self.bins:
                type_match_value = self.type_match(obj, bin_name)
                if binary_to_float:
                    type_match_value = float(type_match_value)
                groundings[f'type_match({obj},{bin_name})'] = type_match_value

        # Smaller predicates
        # for i, obj1 in enumerate(self.objects):
        #     for j, obj2 in enumerate(self.objects):
        #         if i != j:
        #             smaller_value = self.smaller(obj1, obj2)
        #             if binary_to_float:
        #                 smaller_value = float(smaller_value)
        #             groundings[f'smaller({obj1},{obj2})'] = smaller_value

        # Grasped predicates
        for obj in self.objects:
            grasped_value = self.grasped(obj)
            if binary_to_float:
                grasped_value = float(grasped_value)
            groundings[f'grasped({obj})'] = grasped_value

        # On predicates
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i != j:
                    on_value = self.on(obj1, obj2)
                    if binary_to_float:
                        on_value = float(on_value)
                    groundings[f'on({obj1},{obj2})'] = on_value
            for bin_name in self.bins:
                on_value = self.on(obj1, bin_name)
                if binary_to_float:
                    on_value = float(on_value)
                groundings[f'on({obj1},{bin_name})'] = on_value

        # Clear predicates
        for obj in self.objects:
            clear_value = self.clear(obj)
            if binary_to_float:
                clear_value = float(clear_value)
            groundings[f'clear({obj})'] = clear_value
        # Add bin clear predicates
        for bin_name in self.bins:
            groundings[f'clear({bin_name})'] = True

        # Over predicates (gripper over cubes and bins)
        for obj in self.objects + self.bins:
            over_value = self.over('gripper', obj, return_distance=return_distance)
            if return_distance:
                over_value = over_value / self.max_distance
            if binary_to_float:
                over_value = float(over_value)
            groundings[f'over(gripper,{obj})'] = over_value

        # At grab level predicates
        for obj in self.objects:
            at_grab_level_value = self.at_grab_level('gripper', obj, return_distance=return_distance)
            if return_distance and at_grab_level_value is not None:
                at_grab_level_value = at_grab_level_value / self.max_distance
            if binary_to_float:
                at_grab_level_value = float(at_grab_level_value)
            groundings[f'at_grab_level(gripper,{obj})'] = at_grab_level_value

        # In bin predicates
        for obj in self.objects:
            for bin_name in self.bins:
                in_bin_value = self.in_bin(obj, bin_name)
                if binary_to_float:
                    in_bin_value = float(in_bin_value)
                groundings[f'in({obj},{bin_name})'] = in_bin_value

        # Correct bin predicates
        for obj in self.objects:
            correct_value = self.correct_bin(obj)
            if binary_to_float:
                correct_value = float(correct_value)
            groundings[f'correct_bin({obj})'] = correct_value

        # Gripper open
        gripper_open_value = self.open_gripper(return_distance=return_distance)
        if binary_to_float:
            gripper_open_value = float(gripper_open_value)
        groundings['open_gripper(gripper)'] = gripper_open_value

        # Picked up predicates
        for obj in self.objects:
            picked_up_value = self.picked_up(obj, return_distance=return_distance)
            if return_distance:
                picked_up_value = picked_up_value / self.max_distance
            if binary_to_float:
                picked_up_value = float(picked_up_value)
            groundings[f'picked_up({obj})'] = picked_up_value

        return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])

    def dict_to_array(self, groundings):
        return np.asarray([v for k, v in sorted(groundings.items())])


class HeightStackingDetector:
    """
    Detector for HeightStacking environment.
    Task: Stack all cubes on central platform to maximize height.
    """
    def __init__(self, env):
        self.env = env
        self.max_cubes = env.max_cubes
        self.objects = [f'cube{i}' for i in range(self.max_cubes)]
        self.grippers = ['gripper']
        self.max_distance = 10
        self.object_id = {f'cube{i}': env.cubes[i].name+"_main" for i in range(self.max_cubes)}
        self.object_id.update({'platform': 'platform_main'})

    def select_object(self, obj_name):
        """Select cube object by name."""
        for i, cube in enumerate(self.env.cubes):
            if f'cube{i}' == obj_name:
                return cube
        return None

    def grasped(self, obj):
        """Check if object is grasped by gripper."""
        cube_idx = int(obj.replace('cube', ''))
        if cube_idx >= self.env.active_num_cubes:
            return False  # Inactive cube
            
        active_obj = self.env.cubes[cube_idx]
        
        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True
    
    def on(self, obj1, obj2):
        """Check if obj1 is on obj2."""
        if obj2 != 'platform':
            for i, other_cube in enumerate(self.objects):
                if obj2 == other_cube:
                    other_body = self.env.cube_body_ids[i]
                    break
            else:
                return False
            cube_idx = int(obj1.replace('cube', ''))
            if cube_idx >= self.env.active_num_cubes:
                return False
            obj_body = self.env.cube_body_ids[cube_idx]
            obj_pos = self.env.sim.data.body_xpos[obj_body]
            other_pos = self.env.sim.data.body_xpos[other_body]
            dist_xy = np.linalg.norm(obj_pos[:-1] - other_pos[:-1])
            dist_z = other_pos[2] - obj_pos[2]
            if dist_xy < 0.05 and dist_z > 0 and dist_z < 0.05:
                return True
            return False
        else:
            cube_idx = int(obj1.replace('cube', ''))
            if cube_idx >= self.env.active_num_cubes:
                return False
            cube_pos = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]]
            platform_pos = self.env.platform_pos
            return self.env._check_on_platform(cube_pos, platform_pos)
    
    def clear(self, obj):
        """Check if no other cube is on top of the specified cube."""
        cube_idx = int(obj.replace('cube', ''))
        if cube_idx >= self.env.active_num_cubes:
            return False
            
        obj_body = self.env.cube_body_ids[cube_idx]
        obj_pos = self.env.sim.data.body_xpos[obj_body]
        for i, other_cube in enumerate(self.objects):
            if i != cube_idx and i < self.env.active_num_cubes:
                other_body = self.env.cube_body_ids[i]
                other_pos = self.env.sim.data.body_xpos[other_body]
                dist_xy = np.linalg.norm(obj_pos[:-1] - other_pos[:-1])
                dist_z = other_pos[2] - obj_pos[2]
                if dist_xy < 0.05 and dist_z > 0 and dist_z < 0.05:
                    return False
        return True

    def over(self, gripper, obj, return_distance=False):
        """Check if gripper is over object or platform."""
        if gripper != 'gripper':
            return None
            
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
        
        if obj == 'platform':
            obj_pos = self.env.platform_pos
        else:
            cube_idx = int(obj.replace('cube', ''))
            if cube_idx >= self.env.active_num_cubes:
                return self.max_distance if return_distance else False
            obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]])
        
        dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
        
        if return_distance:
            return dist_xy
        else:
            return bool(dist_xy < 0.003)

    def at_grab_level(self, gripper, obj, return_distance=False):
        """Check if gripper is at same height as object."""
        if gripper != 'gripper' or obj == 'platform':
            return None
            
        cube_idx = int(obj.replace('cube', ''))
        if cube_idx >= self.env.active_num_cubes:
            return None
            
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
        obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]])
        dist_z = abs(gripper_pos[2] - obj_pos[2])
        
        if return_distance:
            return dist_z
        else:
            return bool(dist_z < 0.02)

    def on_platform(self, obj):
        """Check if cube is on platform."""
        cube_idx = int(obj.replace('cube', ''))
        if cube_idx >= self.env.active_num_cubes:
            return False
            
        cube_pos = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]]
        return self.env._check_on_platform(cube_pos, self.env.platform_pos)

    def on_cube(self, obj1, obj2):
        """Check if obj1 is stacked on top of obj2."""
        cube_idx1 = int(obj1.replace('cube', ''))
        cube_idx2 = int(obj2.replace('cube', ''))
        
        if cube_idx1 >= self.env.active_num_cubes or cube_idx2 >= self.env.active_num_cubes:
            return False
            
        cube1_pos = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx1]]
        cube2_pos = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx2]]
        
        dist_xy = np.linalg.norm(cube1_pos[:-1] - cube2_pos[:-1])
        dist_z = cube1_pos[2] - cube2_pos[2]
        
        # Check if horizontally aligned, obj1 above obj2, and reasonable z distance
        return bool(dist_xy < 0.03 and 0.04 < dist_z < 0.06)

    def open_gripper(self, return_distance=False):
        """Check if gripper is open."""
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        if return_distance:
            return aperture
        return bool(aperture > 0.13)

    def picked_up(self, obj, return_distance=False):
        """Check if cube has been picked up."""
        cube_idx = int(obj.replace('cube', ''))
        if cube_idx >= self.env.active_num_cubes:
            return self.max_distance if return_distance else False
            
        z_target = self.env.table_offset[2] + 0.4
        object_z_loc = self.env.sim.data.body_xpos[self.env.cube_body_ids[cube_idx]][2]
        z_dist = z_target - object_z_loc
        
        if return_distance:
            return z_dist
        else:
            return bool(z_dist < 0.15)

    def active(self, obj):
        """Check if cube is active in this episode."""
        cube_idx = int(obj.replace('cube', ''))
        return cube_idx < self.env.active_num_cubes

    def smaller(self, obj1, obj2):
        """Check if obj1 is smaller than obj2."""
        cube_idx1 = int(obj1.replace('cube', ''))
        cube_idx2 = int(obj2.replace('cube', ''))
        
        size1 = self.env.cube_sizes[cube_idx1]
        size2 = self.env.cube_sizes[cube_idx2]
        return float(size1) < float(size2)

    def get_all_objects_pos(self):
        # Return a dict of all object positions
        positions = {}
        for obj in self.objects:
            body_id = self.env.sim.model.body_name2id(self.object_id[obj])
            positions[obj] = np.asarray(self.env.sim.data.body_xpos[body_id])
        # Add eef position
        positions['gripper'] = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
        return positions

    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
        """Get all predicates for the environment state."""
        groundings = {}

        # Smaller predicates
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i != j:
                    smaller_value = self.smaller(obj1, obj2)
                    if binary_to_float:
                        smaller_value = float(smaller_value)
                    groundings[f'smaller({obj1},{obj2})'] = smaller_value
        # Platform is bigger than all cubes
        for obj in self.objects:
            groundings[f'smaller({obj},platform)'] = True

        # Active predicates
        for obj in self.objects:
            active_value = self.active(obj)
            if binary_to_float:
                active_value = float(active_value)
            groundings[f'active({obj})'] = active_value

        # Grasped predicates
        for obj in self.objects:
            grasped_value = self.grasped(obj)
            if binary_to_float:
                grasped_value = float(grasped_value)
            groundings[f'grasped({obj})'] = grasped_value

        # On predicates
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i != j:
                    on_value = self.on(obj1, obj2)
                    if binary_to_float:
                        on_value = float(on_value)
                    groundings[f'on({obj1},{obj2})'] = on_value

        # Clear predicates
        for obj in self.objects:
            clear_value = self.clear(obj)
            if binary_to_float:
                clear_value = float(clear_value)
            groundings[f'clear({obj})'] = clear_value

        # Over predicates (gripper over cubes and platform)
        for obj in self.objects + ['platform']:
            over_value = self.over('gripper', obj, return_distance=return_distance)
            if return_distance and over_value != self.max_distance:
                over_value = over_value / self.max_distance
            if binary_to_float:
                over_value = float(over_value)
            groundings[f'over(gripper,{obj})'] = over_value

        # At grab level predicates
        for obj in self.objects:
            at_grab_level_value = self.at_grab_level('gripper', obj, return_distance=return_distance)
            if return_distance and at_grab_level_value is not None:
                at_grab_level_value = at_grab_level_value / self.max_distance
            if binary_to_float and at_grab_level_value is not None:
                at_grab_level_value = float(at_grab_level_value)
            groundings[f'at_grab_level(gripper,{obj})'] = at_grab_level_value if at_grab_level_value is not None else 0.0

        # On platform predicates
        for obj in self.objects:
            on_platform_value = self.on_platform(obj)
            if binary_to_float:
                on_platform_value = float(on_platform_value)
            groundings[f'on({obj},platform)'] = on_platform_value

        # On cube predicates (stacking)
        for obj1 in self.objects:
            for obj2 in self.objects:
                if obj1 != obj2:
                    on_cube_value = self.on_cube(obj1, obj2)
                    if binary_to_float:
                        on_cube_value = float(on_cube_value)
                    groundings[f'on({obj1},{obj2})'] = on_cube_value

        # Gripper open
        gripper_open_value = self.open_gripper(return_distance=return_distance)
        if binary_to_float:
            gripper_open_value = float(gripper_open_value)
        groundings['open_gripper(gripper)'] = gripper_open_value

        # Picked up predicates
        for obj in self.objects:
            picked_up_value = self.picked_up(obj, return_distance=return_distance)
            if return_distance and picked_up_value != self.max_distance:
                picked_up_value = picked_up_value / self.max_distance
            if binary_to_float:
                picked_up_value = float(picked_up_value)
            groundings[f'picked_up({obj})'] = picked_up_value

        return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])

    def dict_to_array(self, groundings):
        return np.asarray([v for k, v in sorted(groundings.items())])


class PatternReplicationDetector:
    """
    Detector for PatternReplication environment.
    Task: Replicate reference pattern on target platform using movable cubes.
    """
    def __init__(self, env):
        self.env = env
        self.num_cubes = env.num_cubes
        self.objects = [f'cube{i}' for i in range(self.num_cubes)]
        self.objects += [f'ref_cube{i}' for i in range(env.num_cubes)]
        self.platforms = ['reference_platform', 'target_platform']
        self.grippers = ['gripper']
        self.max_distance = 10
        self.object_id = {f'cube{i}': env.movable_cubes[i].name+"_main" for i in range(self.num_cubes)}
        self.object_id.update({f'ref_cube{i}': env.reference_cubes[i].name+"_main" for i in range(env.num_cubes)})
        self.object_id.update({'reference_platform': "platform1_main",
                               'target_platform': "target_platform_main"})

    def select_object(self, obj_name):
        """Select movable cube object by name."""
        for i, cube in enumerate(self.env.movable_cubes):
            if f'cube{i}' == obj_name:
                return cube
        return None

    def grasped(self, obj):
        """Check if movable cube is grasped by gripper."""
        if obj not in self.objects or 'ref_cube' in obj:
            return False
        cube_idx = int(obj.replace('cube', ''))
        active_obj = self.env.movable_cubes[cube_idx]
        
        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]
        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True
    
    def on(self, obj1, obj2):
        """Check if obj1 is on obj2.
        obj1 can be a movable cube or a reference cube, obj2 can be a movable cube or platform or reference cube."""
        if obj2 not in self.platforms:
            for i, other_cube in enumerate(self.objects):
                if obj2 == other_cube:
                    if 'ref_cube' in obj2:
                        other_pos = self.env.target_platform_pos + self.env.reference_pattern_positions[int(obj2.replace('ref_cube', ''))]
                    else:
                        other_body = self.env.movable_cube_body_ids[i]
                        other_pos = self.env.sim.data.body_xpos[other_body]
                    break
            else:
                return False
            if 'ref_cube' in obj1:
                obj_pos = self.env.target_platform_pos + self.env.reference_pattern_positions[int(obj1.replace('ref_cube', ''))]
            else:
                cube_idx = int(obj1.replace('cube', ''))
                obj_body = self.env.movable_cube_body_ids[cube_idx]
                obj_pos = self.env.sim.data.body_xpos[obj_body]

            dist_xy = np.linalg.norm(obj_pos[:-1] - other_pos[:-1])
            dist_z = obj_pos[2] - other_pos[2]
            if dist_xy < 0.05 and dist_z > 0 and dist_z < self.env.cube_size*3:
                return True
            return False
        else:
            if 'ref_cube' in obj1:
                cube_pos = self.env.target_platform_pos + self.env.reference_pattern_positions[int(obj1.replace('ref_cube', ''))]
            else:
                cube_idx = int(obj1.replace('cube', ''))
                cube_pos = self.env.sim.data.body_xpos[self.env.movable_cube_body_ids[cube_idx]]
            if obj2 == 'reference_platform':
                platform_pos = self.env.reference_platform_pos
            else:
                platform_pos = self.env.target_platform_pos
            return self.env._check_on_platform(cube_pos, platform_pos)

    def clear(self, obj):
        """Check if no other cube is on top of the specified cube."""
        if 'ref_cube' in obj:
            # Reference cubes are static, check if any movable cube is on top
            ref_cube_idx = int(obj.replace('ref_cube', ''))
            obj_pos = self.env.target_platform_pos + self.env.reference_pattern_positions[ref_cube_idx]
            
            for i in range(self.env.num_cubes):
                other_body = self.env.movable_cube_body_ids[i]
                other_pos = self.env.sim.data.body_xpos[other_body]
                dist_xy = np.linalg.norm(obj_pos[:-1] - other_pos[:-1])
                dist_z = other_pos[2] - obj_pos[2]
                if dist_xy < 0.05 and dist_z > 0 and dist_z < 0.05:
                    return False
            return True
        else:
            # Movable cube
            cube_idx = int(obj.replace('cube', ''))
            obj_body = self.env.movable_cube_body_ids[cube_idx]
            obj_pos = self.env.sim.data.body_xpos[obj_body]
            
            # Check if any other movable cube is on top
            for i in range(self.env.num_cubes):
                if i != cube_idx:
                    other_body = self.env.movable_cube_body_ids[i]
                    other_pos = self.env.sim.data.body_xpos[other_body]
                    dist_xy = np.linalg.norm(obj_pos[:-1] - other_pos[:-1])
                    dist_z = other_pos[2] - obj_pos[2]
                    if dist_xy < 0.05 and dist_z > 0 and dist_z < 0.05:
                        return False
            return True
    
    def over(self, gripper, obj, return_distance=False):
        """Check if gripper is over movable cube or platform."""
        if "ref_cube" in obj:
            return None
        if gripper != 'gripper':
            return None
            
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
        
        if obj in self.platforms:
            # Get platform position
            if obj == 'reference_platform':
                obj_pos = self.env.reference_platform_pos
            else:
                obj_pos = self.env.target_platform_pos
        else:
            # Get movable cube position
            cube_idx = int(obj.replace('cube', ''))
            obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.movable_cube_body_ids[cube_idx]])
        dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
        if return_distance:
            return dist_xy
        else:
            return bool(dist_xy < 0.002)
        
    def at_grab_level(self, gripper, obj, return_distance=False):
        """Check if gripper is at same height as movable cube."""
        if gripper != 'gripper' or obj in self.platforms or "ref_cube" in obj:
            return None
            
        cube_idx = int(obj.replace('cube', ''))
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id('gripper0_eef')])
        obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.movable_cube_body_ids[cube_idx]])
        dist_z = abs(gripper_pos[2] - obj_pos[2])
        if return_distance:
            return dist_z
        else:
            return bool(dist_z < 0.02)
        
    def on_platform(self, obj, platform):
        """Check if movable cube is on specified platform."""
        if "ref_cube" in obj:
            return False
        cube_idx = int(obj.replace('cube', ''))
        cube_pos = self.env.sim.data.body_xpos[self.env.movable_cube_body_ids[cube_idx]]
        
        if platform == 'reference_platform':
            platform_pos = self.env.reference_platform_pos
        else:
            platform_pos = self.env.target_platform_pos
        
        return self.env._check_on_platform(cube_pos, platform_pos)
    
    def open_gripper(self, return_distance=False):
        """Check if gripper is open."""
        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
        if return_distance:
            return aperture
        return bool(aperture > 0.13)
    
    def picked_up(self, obj, return_distance=False):
        """Check if movable cube has been picked up."""
        if "ref_cube" in obj:
            return False
        cube_idx = int(obj.replace('cube', ''))
        z_target = self.env.table_offset[2] + 0.25
        object_z_loc = self.env.sim.data.body_xpos[self.env.movable_cube_body_ids[cube_idx]][2]
        z_dist = z_target - object_z_loc
        if return_distance:
            return z_dist
        else:
            return bool(z_dist < 0.1)

    def get_all_objects_pos(self):
        # Return a dict of all object positions
        positions = {}
        for obj in self.objects:
            body_id = self.env.sim.model.body_name2id(self.object_id[obj])
            positions[obj] = np.asarray(self.env.sim.data.body_xpos[body_id])
        # Add eef position
        positions['gripper'] = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
        return positions

        
    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
        """Get all predicates for the environment state."""
        groundings = {}

        # Grasped predicates
        for obj in self.objects:
            grasped_value = self.grasped(obj)
            if binary_to_float:
                grasped_value = float(grasped_value)
            groundings[f'grasped({obj})'] = grasped_value

        # On predicates
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i != j:
                    on_value = self.on(obj1, obj2)
                    if binary_to_float:
                        on_value = float(on_value)
                    groundings[f'on({obj1},{obj2})'] = on_value

        # Clear predicates
        for obj in self.objects:
            clear_value = self.clear(obj)
            if binary_to_float:
                clear_value = float(clear_value)
            groundings[f'clear({obj})'] = clear_value
        # Add platform clear predicates
        for platform in self.platforms:
            groundings[f'clear({platform})'] = True

        # Over predicates (gripper over movable cubes and platforms)
        for obj in self.objects + self.platforms:
            over_value = self.over('gripper', obj, return_distance=return_distance)
            if return_distance:
                over_value = over_value / self.max_distance
            if binary_to_float:
                over_value = float(over_value)
            groundings[f'over(gripper,{obj})'] = over_value

        # At grab level predicates
        for obj in self.objects:
            at_grab_level_value = self.at_grab_level('gripper', obj, return_distance=return_distance)
            if return_distance:
                at_grab_level_value = at_grab_level_value / self.max_distance
            if binary_to_float:
                at_grab_level_value = float(at_grab_level_value)
            groundings[f'at_grab_level(gripper,{obj})'] = at_grab_level_value

        # On platform predicates
        for obj in self.objects:
            for platform in self.platforms:
                on_platform_value = self.on_platform(obj, platform)
                if binary_to_float:
                    on_platform_value = float(on_platform_value)
                groundings[f'on({obj},{platform})'] = on_platform_value

        # Gripper open
        gripper_open_value = self.open_gripper(return_distance=return_distance)
        if binary_to_float:
            gripper_open_value = float(gripper_open_value)
        groundings['open_gripper(gripper)'] = gripper_open_value

        # Picked up predicates
        for obj in self.objects:
            picked_up_value = self.picked_up(obj, return_distance=return_distance)
            if return_distance:
                picked_up_value = picked_up_value / self.max_distance
            if binary_to_float:
                picked_up_value = float(picked_up_value)
            groundings[f'picked_up({obj})'] = picked_up_value

        return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])
    
    def dict_to_array(self, groundings):
        return np.asarray([v for k, v in sorted(groundings.items())])
    
    def get_pattern_replication_goal(self):
        """Get goal state for pattern replication task define 
        with 'on' predicates between reference cubes and target platform."""
        groundings = {}
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i != j:
                    on_value = self.on(obj1, obj2)
                    groundings[f'on({obj1},{obj2})'] = on_value
        for key in list(groundings.keys()):
            if 'ref_cube' not in key or groundings[key] is False:
                del groundings[key]
        goal_list = list(groundings.keys())
        # Replace ref_cube with cube to indicate goal
        for i in range(len(goal_list)):
            goal_list[i] = goal_list[i].replace('ref_cube', 'cube')
        # For the cube that is on nothing, add on(cube, platform)
        cubes_on_nothing = set()
        for i in range(self.env.num_cubes):
            cube_name = f'cube{i}'
            on_anything = False
            for goal in goal_list:
                if goal.startswith(f'on({cube_name},'):
                    on_anything = True
                    break
            if not on_anything:
                cubes_on_nothing.add(cube_name)
        for cube_name in cubes_on_nothing:
            goal_list.append(f'on({cube_name},target_platform)')
        # replace the format on(obj1,obj2) with (on obj1 obj2)
        for i in range(len(goal_list)):
            goal = goal_list[i]
            goal = goal.replace('on(', 'on ')
            goal = goal.replace(',', ' ')
            goal = goal.replace(')', '')
            goal_list[i] = goal
        return goal_list
    


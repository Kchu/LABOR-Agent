#!/usr/bin/env python
"""
Implementation of Large Language Model Coordinator for NICOL Bimanual Robot
Author: Kun Chu (kun.chu@uni-hamburg.de)
Copyright 2024, Planet Earth
"""
import numpy as np
from transforms3d.euler import euler2quat, quat2euler

from nicol_api.nicol_env import NicolFactory
from nicol_api.base import NicolPose
from coppeliasim_zmqremoteapi_client import *

import argparse, json, time
import scipy.spatial.distance as dist

############################ Initialize ######################################
nicol = NicolFactory().create_nicol('coppelia', scene="./nicol.ttt", start_scene=False, talker=False)
head  = nicol.head()    # 3D-Printed head structure
left  = nicol.left()    # Left  OpenManipulator + RH8D
right = nicol.right()   # Right OpenManipulator + RH8D
sim = nicol.nicol_adapter.sim
left_sensor = sim.getObject('/l_sensor')
left_attachpoint = sim.getObject('/l_palm_attachPoint')
right_sensor = sim.getObject('/r_sensor')
right_attachpoint = sim.getObject('/r_palm_attachPoint')
client = nicol.nicol_adapter.client
# print("\n"*30, "#"*114, '\n', "#"*114, '\n', 'The simulation starts!')

# Set up useful dicts and global varibles.
##############################################################################
ORIENTATION_DICT_LEFT = {
    'Horizontally_Up': euler2quat(0, np.pi/2, np.pi/2),
    'Horizontally_Down': euler2quat(np.pi, -np.pi/2, np.pi/2),
    'Vertical': euler2quat(np.pi/2, np.pi, np.pi),
    'Horizontally_Slanted_Up': euler2quat(2*np.pi/3, 5*np.pi/6, np.pi),
    'Horizontally_Slanted_Down': euler2quat(-np.pi, -1*np.pi/4, np.pi*5/8),
    'Vertically_Slanted': euler2quat(np.pi/2, np.pi, np.pi*4/3),
    'Vertically_Down': euler2quat(8*np.pi/8, -np.pi/8, 4*np.pi/8)}
ORIENTATION_DICT_RIGHT = {
    'Horizontally_Up': euler2quat(0, np.pi/2, np.pi/2),
    'Horizontally_Down': euler2quat(np.pi, -np.pi/2, np.pi/2),
    'Vertical': euler2quat(np.pi/2, 0, np.pi),
    'Horizontally_Slanted_Up': euler2quat(2*np.pi/8, np.pi/6, np.pi),
    'Horizontally_Slanted_Down': euler2quat(np.pi, -np.pi*6/8, np.pi*5/8),
    'Vertically_Slanted': euler2quat(np.pi/2, 0, np.pi*2/3),
    'Vertically_Down': euler2quat(8*np.pi/8, 0, 3*np.pi/8)}
RIGHT_FINGER_STAT, RIGHT_HAND_STAT = 'Open', 'Vertical'
LEFT_FINGER_STAT, LEFT_HAND_STAT = 'Open', 'Vertical'
##############################################################################

def reset_global():
    global RIGHT_FINGER_STAT
    global LEFT_FINGER_STAT
    RIGHT_FINGER_STAT = LEFT_FINGER_STAT = 'Open'
    global LEFT_HAND_STAT
    global RIGHT_HAND_STAT
    LEFT_HAND_STAT = RIGHT_HAND_STAT = 'Vertical'

# NICOL Skill Functions 
##############################################################################

def side_grasp(side: str, obj_name: str):
    global LEFT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_HAND_STAT, RIGHT_FINGER_STAT
    if side == 'left':
        hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        hand_finger_info = ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    if "Bowl" in obj_name:
        return f"The robot's {side} hand failed to grasp the bowl." + hand_finger_info
    elif "overlap_area" in obj_name:
        return f"Overlap area is not grasable! You should choose specific object." + hand_finger_info
    handle = sim.getObject('/'+obj_name)
    new_pose = sim.getObjectPosition(handle, sim.handle_world)
    # head.set_pose_target(NicolPose(new_pose, [0, 0, 0, 0]))
    if side == 'left':
        detected = sim.checkProximitySensor(left_sensor, sim.handle_all)[0]
        # grasped
        if LEFT_FINGER_STAT == 'Closed' and detected:
            return f"The {side} hand failed to grasp {obj_name}, as it is already occupied." + hand_finger_info
        # out of area
        if new_pose[1]<-0.2: 
            return f"The {side} hand failed to grasp {obj_name}, out of its area." + hand_finger_info
        # move
        result = move_single_to_pose(side, obj_name, 'Vertical', off_set=None)
        if 'failed' in result:
            return result + hand_finger_info
    elif side == 'right':
        detected = sim.checkProximitySensor(right_sensor, sim.handle_all)[0]
        if RIGHT_FINGER_STAT == 'Closed' and detected:
            return f"The {side} hand failed to grasp {obj_name}, as it is already occupied." + hand_finger_info
        if new_pose[1]>0.2: 
            return f"The {side} hand failed to grasp {obj_name}, out of its area." + hand_finger_info
        result = move_single_to_pose(side, obj_name, 'Vertical', off_set=None)
        if 'failed' in result:
            return result + hand_finger_info
    else:    
        return "the grasp is failed, as you didn't choose the correct side for NICOL."
    quat = [[np.pi, - np.pi, - np.pi, - np.pi, - np.pi], 
            [np.pi, - np.pi, - np.pi, - np.pi, - np.pi], 
            [np.pi, -2.9, -1.8, -1.8, -1.8]]
            # [np.pi, - 2.8, - 1.4, - 1.4, - 1.4]]
    if side == 'left':
        Sensor = left_sensor
        AttachPoint = left_attachpoint
        Controller = left
        LEFT_HAND_STAT = 'Vertical'
        LEFT_FINGER_STAT = 'Closed'
    else:
        Sensor = right_sensor
        AttachPoint = right_attachpoint
        Controller = right
    detected = sim.checkProximitySensor(Sensor, sim.handle_all)[0]
    if detected:
        read_result = sim.readProximitySensor(Sensor)
        detected_handle = read_result[3]
        sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(detected_handle, AttachPoint, True)
        obj_name_grasped = sim.getObjectAlias(read_result[3], -1)
        if '_respondable' in obj_name_grasped:
            obj_name_grasped = obj_name_grasped.replace('_respondable', '')
        for sub_quat in quat:
            Controller.set_joint_position_for_hand(sub_quat, block=True)
        if side == 'left':
            LEFT_HAND_STAT, LEFT_FINGER_STAT = 'Vertical', 'Closed'
        else:
            RIGHT_HAND_STAT, RIGHT_FINGER_STAT = 'Vertical', 'Closed'
        result = f"The robot's {side} hand has grasped {obj_name_grasped}, and is holding the {obj_name_grasped}."
    else:
        result = f"The robot's {side} hand failed to grasp anything, as there is no object detected."
    if side == 'left':
        hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        hand_finger_info = ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    return result + hand_finger_info

def top_grasp(side: str, obj_name: str):
    global LEFT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_HAND_STAT, RIGHT_FINGER_STAT
    if side == 'left':
        hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        hand_finger_info = ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    if "Bowl" in obj_name:
        return f"The robot's {side} hand failed to grasp the bowl." + hand_finger_info
    handle = sim.getObject('/'+obj_name)
    new_pose = sim.getObjectPosition(handle, sim.handle_world)
    # head.set_pose_target(NicolPose(new_pose, [0, 0, 0, 0]))
    if side == 'left':
        if sim.checkProximitySensor(right_sensor, handle)[0]:
            return f"The {side} hand failed to grasp {obj_name}, as the {obj_name} is grasped by the other hand." + hand_finger_info
        detected = sim.checkProximitySensor(left_sensor, sim.handle_all)[0]
        if LEFT_FINGER_STAT == 'Closed' and detected:
            return f"The {side} hand failed to grasp {obj_name}, as the left hand is already occupied." + hand_finger_info
        if LEFT_FINGER_STAT == 'Hold_Up':
            release('left')
        if new_pose[1]<-0.2: 
            return f"The {side} hand failed to grasp {obj_name}, out of its area." + hand_finger_info
        result = move_single_to_pose(side, obj_name, 'Horizontally_Down', off_set=None)
        if 'failed' in result: return result + hand_finger_info
    elif side == 'right':
        if sim.checkProximitySensor(left_sensor, handle)[0]:
            return f"The {side} hand failed to grasp {obj_name}, as the {obj_name} is grasped by the other hand." + hand_finger_info
        detected = sim.checkProximitySensor(right_sensor, sim.handle_all)[0]
        if RIGHT_FINGER_STAT == 'Closed' and detected:
            return "the grasp is failed, as the right hand is already occupied." + hand_finger_info
        if RIGHT_FINGER_STAT == 'Hold_Up':
            release('right')
        if new_pose[1]>0.2: 
            return f"The {side} hand failed to grasp {obj_name}, out of its area." + hand_finger_info
        result = move_single_to_pose(side, obj_name, 'Horizontally_Down', off_set=None)
        if 'failed' in result: return result + hand_finger_info

    quat = [[np.pi, - np.pi, - np.pi, - np.pi, - np.pi],
            [np.pi, - np.pi/2, - 0.3, 0, 0],
            [np.pi, - 2, - 0.3, 0, 0]]
    if side == 'left':
        Sensor, AttachPoint, Controller = left_sensor, left_attachpoint, left
    else:
        Sensor, AttachPoint, Controller = right_sensor, right_attachpoint, right
    detected = sim.checkProximitySensor(Sensor, sim.handle_all)[0]
    # attach the graspable object
    if detected:
        read_result = sim.readProximitySensor(Sensor)
        detected_handle = read_result[3]
        sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(detected_handle, AttachPoint, True)
        obj_name_grasped = sim.getObjectAlias(read_result[3], -1)
        if '_respondable' in obj_name_grasped:
            obj_name_grasped = obj_name_grasped.replace('_respondable', '')
        if obj_name_grasped != 'Bowl':
            for sub_quat in quat:
                Controller.set_joint_position_for_hand(sub_quat)
            result = f"The robot's {side} hand has grasped {obj_name_grasped} from the top, and is lifting the {obj_name_grasped}."
        if side == 'left':
            LEFT_HAND_STAT, LEFT_FINGER_STAT = 'Horizontally_Down', 'Closed'
        else:
            RIGHT_HAND_STAT, RIGHT_FINGER_STAT = 'Horizontally_Down', 'Closed'
    else:
        result = f"The robot's {side} hand failed to grasp anything from the top, as there is no appropriate object detected."
    # return result
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT

def move_single_to_pose(side: str, obj_name: str, ori_angle = None, off_set=None):
    global LEFT_FINGER_STAT, LEFT_HAND_STAT, RIGHT_FINGER_STAT, RIGHT_HAND_STAT
    hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT if side == 'left' else ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    try:
        handle = sim.getObject('/'+obj_name)
        new_pose = sim.getObjectPosition(handle, sim.handle_world)
    except Exception as e:
        return f"The robot's {side} hand failed to move to {obj_name} as the {obj_name} is not detected in the scene." + hand_finger_info
    grasped_object = ' '
    if side == 'left':
        NICOL_Dict = ORIENTATION_DICT_LEFT
        CONTROLLER, CONTROLLER_Offset, CONTROLLER_HAND_STAT, CONTROLLER_FINGER_STAT = left, '_left', LEFT_HAND_STAT, LEFT_FINGER_STAT
        detected = sim.checkProximitySensor(left_sensor, sim.handle_all)[0]
        # out of area
        if new_pose[1]<-0.2: 
            return f"The robot's {side} hand failed to move to {obj_name}, out of its area." + hand_finger_info
        if LEFT_FINGER_STAT == 'PointAt':
            new_pose = [new_pose[0]-0.03, new_pose[1]+0.05, new_pose[2]+0.09]
        elif LEFT_FINGER_STAT == 'Closed':
            if detected:
                read_result = sim.readProximitySensor(left_sensor)
                detected_handle = read_result[3]
                detected_object = sim.getObjectAlias(detected_handle, -1)
                if '_respondable' in detected_object:
                    detected_object = detected_object.replace('_respondable', '')
                grasped_object = ' with ' + detected_object + ' grasped in the hand'
                if obj_name in ('blue_cup', 'yellow_cup'):
                    if off_set == None or off_set == 'None':
                        return f"The robot's {side} hand failed to move to {obj_name}, as the {side} hand is holding {detected_object}." + hand_finger_info
            else:
                grasped_object = ''
            # move above bowl
            # overlap area
            if obj_name == 'overlap_area':
                if off_set == 'up':
                    new_pose = [new_pose[0], new_pose[1]+0.06, new_pose[2]]
                else:
                    new_pose = [new_pose[0], new_pose[1]+0.1, new_pose[2]]
            # offset because of the grasped object
            else:
                new_pose = [new_pose[0], new_pose[1]+0.04, new_pose[2]]
        elif LEFT_FINGER_STAT == 'Open':
            if obj_name == 'overlap_area':
                new_pose = [new_pose[0], new_pose[1]+0.02, new_pose[2]]
            elif 'Origin' in obj_name:
                new_pose = [new_pose[0], new_pose[1]+0.05, new_pose[2]]
        elif detected and LEFT_FINGER_STAT == 'Hold_Up':
            if obj_name == 'overlap_area':
                read_result = sim.readProximitySensor(left_sensor)
                detected_object = sim.getObjectAlias(read_result[3], -1)
                return push_to(side, detected_object, obj_name)
            else:
                return f"The robot's {side} hand failed to move to {obj_name}, as it is holding up some object." + hand_finger_info
        if LEFT_HAND_STAT == 'Horizontally_Down' and obj_name == 'Bowl':
            off_set= 'up'
    elif side == 'right':
        NICOL_Dict = ORIENTATION_DICT_RIGHT
        CONTROLLER, CONTROLLER_Offset, CONTROLLER_HAND_STAT, CONTROLLER_FINGER_STAT = right, '_right', RIGHT_HAND_STAT, RIGHT_FINGER_STAT
        detected = sim.checkProximitySensor(right_sensor, sim.handle_all)[0]
        if new_pose[1]>0.2: 
            return f"The robot's {side} hand failed to move to {obj_name}, out of its area." + hand_finger_info
        if RIGHT_FINGER_STAT == 'PointAt':
            new_pose = [new_pose[0]-0.03, new_pose[1]-0.05, new_pose[2]+0.09]
        elif RIGHT_FINGER_STAT == 'Closed':
            if detected:
                read_result = sim.readProximitySensor(right_sensor)
                detected_handle = read_result[3]
                detected_object = sim.getObjectAlias(detected_handle, -1)
                if '_respondable' in detected_object:
                    detected_object = detected_object.replace('_respondable', '')
                grasped_object = ' with ' + detected_object + ' grasped in the hand'
                if obj_name in ('blue_cup', 'yellow_cup'):
                    if off_set == None or off_set == 'None':
                        return f"The robot's {side} hand failed to move to {obj_name}, as the {side} hand is holding {detected_object}." + hand_finger_info
            else:
                grasped_object = ''
            if obj_name == 'overlap_area':
                if off_set == 'up':
                    new_pose = [new_pose[0], new_pose[1]-0.06, new_pose[2]]
                else:
                    new_pose = [new_pose[0], new_pose[1]-0.1, new_pose[2]]
            else:
                new_pose = [new_pose[0], new_pose[1]-0.04, new_pose[2]]
        elif RIGHT_FINGER_STAT == 'Open':
            if obj_name == 'overlap_area':
                new_pose = [new_pose[0], new_pose[1]-0.02, new_pose[2]]
            elif 'Origin' in obj_name:
                new_pose = [new_pose[0], new_pose[1]-0.05, new_pose[2]]
        elif detected and RIGHT_FINGER_STAT == 'Hold_Up':
            if obj_name == 'overlap_area':
                read_result = sim.readProximitySensor(right_sensor)
                detected_object = sim.getObjectAlias(read_result[3], -1)
                return push_to(side, detected_object, obj_name)
            else:
                return f"The robot's {side} hand failed to move to {obj_name}, as it is holding up some object." + hand_finger_info
        if RIGHT_HAND_STAT == 'Horizontally_Down' and obj_name == 'Bowl':
            off_set= 'up'
    else:
        print("You didn't choose the correct side for NICOL!")
    
    if ori_angle:
        new_quat = NICOL_Dict[ori_angle]
        if ori_angle=='Vertical':
            try: 
                handle = sim.getObject('/'+obj_name + CONTROLLER_Offset)
                new_pose = sim.getObjectPosition(handle, sim.handle_world)
            except Exception as e:
                pass
        elif ori_angle=='Horizontally_Down':
            try: 
                handle = sim.getObject('/'+obj_name+'_top')
                new_pose = sim.getObjectPosition(handle, sim.handle_world)
            except Exception as e: 
                new_pose = [new_pose[0], new_pose[1], new_pose[2]+0.015]
    else:
        new_quat = CONTROLLER.get_eef_pose().orientation
        new_quat = [new_quat.x, new_quat.y, new_quat.z, new_quat.w]
        if CONTROLLER_HAND_STAT == 'Vertical':
            try: 
                handle = sim.getObject('/' + obj_name + CONTROLLER_Offset)
                new_pose = sim.getObjectPosition(handle, sim.handle_world)
            except Exception as e:
                pass
        elif CONTROLLER_HAND_STAT=='Horizontally_Down':
            try: 
                handle = sim.getObject('/' + obj_name + '_top')
                new_pose = sim.getObjectPosition(handle, sim.handle_world)
            except Exception as e: 
                new_pose = [new_pose[0], new_pose[1], new_pose[2]+0.015]

    if off_set in ('up', 'above'):
        new_pose_w_offset = [[new_pose[0], new_pose[1], new_pose[2]+0.18]]
    elif off_set in ('down', 'below'):
        new_pose_w_offset = [[new_pose[0], new_pose[1], new_pose[2]-0.18]]
    elif off_set == 'left':
        new_pose_w_offset = [[new_pose[0], new_pose[1]+0.06, new_pose[2]]]
    elif off_set == 'right':
        new_pose_w_offset = [[new_pose[0], new_pose[1]-0.06, new_pose[2]]]
    else:
        new_pose_w_offset = [[new_pose[0], new_pose[1], new_pose[2]]]
    try:
        if off_set:
            CONTROLLER.set_pose_target(NicolPose(new_pose_w_offset[0], new_quat))
        else:
            EE_POSE = CONTROLLER.get_eef_pose().position.as_list()
            new_pose_offset = new_pose.copy()
            if side == 'left':
                new_pose_offset = [new_pose_offset[0], new_pose_offset[1]+0.03, new_pose_offset[2]]
            elif side == 'right':
                new_pose_offset = [new_pose_offset[0], new_pose_offset[1]-0.03, new_pose_offset[2]]
            CONTROLLER.set_pose_target(NicolPose(new_pose_offset, new_quat))
            CONTROLLER.set_pose_target(NicolPose(new_pose, new_quat))
        if obj_name == 'serve_point':
            head.set_pose_target(NicolPose([new_pose[0], new_pose[1], new_pose[2]+0.1], [0, 0, 0, 0]))
        if off_set == 'up':
            result = f"The robot's {side} hand has moved above the {obj_name}{grasped_object}."
        elif off_set == 'down':
            result = f"The robot's {side} hand has moved below the {obj_name}{grasped_object}."
        else:
            result = f"The robot's {side} hand has moved to {obj_name}{grasped_object}."
        if side == 'left':
            return result + hand_finger_info
        else:
            return result + hand_finger_info
        # return result
    except Exception as e:
        print("You didn't choose the correct orientation for the robot!")
        return e
    
def delta_move(side, direction, distance=None):
    if side == 'left':
        CONTROLLER = left
    elif side == 'right':
        CONTROLLER = right
    EE_POSE = CONTROLLER.get_eef_pose().position.as_list()
    ori = CONTROLLER.get_eef_pose().orientation
    ori = [ori.x, ori.y, ori.z, ori.w]
    # ori = [ori.w, ori.x, ori.y, ori.z]
    if distance:
        distance = min(float(distance), 0.08)
    else:
        distance = 0.03
    if direction in ['right', 'y-decrease']:
        new_pose = [EE_POSE[0], EE_POSE[1]-distance, EE_POSE[2]]
    elif direction in ['left', 'y-increase']:
        new_pose = [EE_POSE[0], EE_POSE[1]+distance, EE_POSE[2]]
    elif direction in ['front', 'x-increase']:
        new_pose = [EE_POSE[0]+distance, EE_POSE[1], EE_POSE[2]]
    elif direction in ['behind', 'back', 'x-decrease']:
        new_pose = [EE_POSE[0]-distance, EE_POSE[1], EE_POSE[2]]
    elif direction == 'down':
        new_pose = [EE_POSE[0], EE_POSE[1], EE_POSE[2]-distance]
    elif direction == 'up':
        new_pose = [EE_POSE[0], EE_POSE[1], EE_POSE[2]+distance]
    else: new_pose = [EE_POSE[0], EE_POSE[1], EE_POSE[2]]
    CONTROLLER.set_pose_target(NicolPose(new_pose, ori))
    result = f"The robot's {side} hand has moved {direction} for {distance} meters."
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    # return result

def release(side:str):
    global LEFT_FINGER_STAT, LEFT_HAND_STAT, RIGHT_FINGER_STAT, RIGHT_HAND_STAT
    if side == 'left':
        CONTROLLER, CONTROLLER_Sensor = left, left_sensor
        CONTROLLER_DICT = ORIENTATION_DICT_LEFT
        LEFT_FINGER_STAT = "Open"
        CONTROLLER_Ori = LEFT_HAND_STAT
        delta_move('left', 'right', 0.02)
    elif side == 'right':
        CONTROLLER, CONTROLLER_Sensor = right, right_sensor
        CONTROLLER_DICT = ORIENTATION_DICT_RIGHT
        RIGHT_FINGER_STAT = "Open"
        CONTROLLER_Ori = RIGHT_HAND_STAT
        delta_move('right', 'left', 0.02)
    else:
        print("You didn't choose the correct side for NICOL!")
    detected = sim.checkProximitySensor(CONTROLLER_Sensor, sim.handle_all)[0]
    if detected:
        read_result = sim.readProximitySensor(CONTROLLER_Sensor)
        detected_handle = read_result[3]
        sim.setObjectParent(detected_handle, -1, True)
        if CONTROLLER_Ori == 'Horizontally_Down':
            sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 0)
            time.sleep(1)
            sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 0)
            time.sleep(1)
            sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 0)
        obj_name = sim.getObjectAlias(detected_handle, -1)
        if '_respondable' in obj_name:
            obj_name = obj_name.replace('_respondable', '')
        # sim.step()
    quat = [-np.pi, - np.pi, - np.pi, - np.pi, - np.pi]
    CONTROLLER.set_joint_position_for_hand(quat)
    # set up simulated step to enable the object fall down
    time.sleep(1)
    client.setStepping(True)
    client.step()
    time.sleep(1)
    client.step()
    client.setStepping(False)
    if side == 'left' and LEFT_HAND_STAT == 'Vertical':
        delta_move('left', 'up', 0.06)
        delta_move('left', 'left', 0.03)
        # new_pose = [EE_POSE[0], EE_POSE[1]+0.03, EE_POSE[2]+0.06]
    elif side == 'right' and RIGHT_HAND_STAT == 'Vertical':
        delta_move('right', 'up', 0.06)
        delta_move('right', 'right', 0.03)
        # new_pose = [EE_POSE[0], EE_POSE[1]-0.03, EE_POSE[2]+0.06]
    if side == 'right' and RIGHT_HAND_STAT == 'Horizontally_Down':
        delta_move('right', 'up', 0.08)
        delta_move('right', 'up', 0.08)
        # new_pose = [EE_POSE[0], EE_POSE[1], EE_POSE[2]+0.08]
    elif side == 'left' and LEFT_HAND_STAT == 'Horizontally_Down':
        delta_move('left', 'up', 0.08)
        delta_move('left', 'up', 0.0)
    if detected:
        result = f"The robot's {side} hand has been opened, and the grasped {obj_name} has been released."
    else:
        result = f"The robot's {side} hand has been opened."
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT


def reset(side:str):
    if side == 'left':
        release('left')
        left.set_joint_position_for_hand([-np.pi] * 5, block=True)
        left.set_joint_position([-1.57] + [0.] * 7, block=True)
        global LEFT_FINGER_STAT, LEFT_HAND_STAT
        LEFT_FINGER_STAT, LEFT_HAND_STAT = 'Open', 'Vertical'
        result = f"The robot's {side} hand has been reset to its original position."
    elif side == 'right':
        release('right')
        right.set_joint_position_for_hand([-np.pi] * 5, block=True)
        right.set_joint_position([1.57] + [0.] * 7, block=True)
        global RIGHT_FINGER_STAT, RIGHT_HAND_STAT
        RIGHT_FINGER_STAT, RIGHT_HAND_STAT = 'Open', 'Vertical'
        result = f"The robot's {side} hand has been reset to its original position."
    else:
        return "You didn't choose the correct side for NICOL!"
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    # return result

def flip_down(side:str):
    try:
        ball = sim.getObject("/big_ball")
    except Exception as e:
        return 'There is no water inside the container!'
    if side == 'left':
        CONTROLLER = left
        global LEFT_HAND_STAT
        # LEFT_HAND_STAT = 'Horizontally_Down'
        CONTROLLER_sensor = left_sensor
        CONTROLLER_DICT = ORIENTATION_DICT_LEFT
        EE_POSE = CONTROLLER.get_eef_pose().position.as_list()
        new_pose_1 = [EE_POSE[0]+0.01, EE_POSE[1]+0.01, EE_POSE[2]]
        # if  sim.checkProximitySensor(CONTROLLER_sensor, ball)[0]:
        sim.setObjectParent(ball, -1, True)
        sim.setObjectInt32Parameter(ball, sim.shapeintparam_static, 0)
    elif side == 'right':
        CONTROLLER = right
        global RIGHT_HAND_STAT
        # RIGHT_HAND_STAT = 'Horizontally_Down'
        CONTROLLER_DICT = ORIENTATION_DICT_RIGHT
        CONTROLLER_sensor = right_sensor
        EE_POSE = CONTROLLER.get_eef_pose().position.as_list()
        new_pose_1 = [EE_POSE[0]+0.01, EE_POSE[1]-0.02, EE_POSE[2]]
        # if  sim.checkProximitySensor(CONTROLLER_sensor, ball)[0]:
        sim.setObjectParent(ball, -1, True)
        sim.setObjectInt32Parameter(ball, sim.shapeintparam_static, 0)
    else:
        print("You didn't choose the correct side for NICOL!")
    # head.set_pose_target(NicolPose([EE_POSE[0], EE_POSE[1], EE_POSE[2]+0.1], [0, 0, 0, 0]))
    ori_1 = CONTROLLER_DICT['Horizontally_Slanted_Down']
    CONTROLLER.set_pose_target(NicolPose(new_pose_1, ori_1))
    CONTROLLER.set_pose_target(NicolPose(new_pose_1, ori_1))
    ori_2= CONTROLLER_DICT['Vertical']
    new_pose_2 = [new_pose_1[0], new_pose_1[1], new_pose_1[2]+ 0.03]
    # return [[new_pose_1, ori_1], [new_pose_2, ori_2]]
    CONTROLLER.set_pose_target(NicolPose(new_pose_2, ori_2))
    result = f"The robot's {side} hand's is fliped down for once for pouring out water."
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    # return result


def hold_up_single(side:str, obj_name:str):
    global LEFT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_HAND_STAT, RIGHT_FINGER_STAT
    quat = [-np.pi, -np.pi, -np.pi*0.75, -np.pi*0.75, -np.pi*0.75]
    obj_pose = sim.getObjectPosition(sim.getObject('/'+obj_name), sim.handle_world)
    obj_pose = [obj_pose[0], obj_pose[1], obj_pose[2]+0.18]
    # head.set_pose_target(obj_pose)
    if side == 'left':
        detected = sim.checkProximitySensor(left_sensor, sim.handle_all)[0]
        if 'cup' in obj_name:
            return "hold_up is failed, as the chosed object is not suitable to hold up." + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
        left_target_pose = sim.getObjectPosition(sim.getObject('/'+obj_name +'_left'), sim.handle_world)
        if obj_pose[1]<-0.25:
            return "hold_up is failed, out of its area." + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
        if LEFT_FINGER_STAT == 'Closed' and detected:
            return "The left hand is already occupied for grasping and holding some object." + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
        else:
            release('left')
        if LEFT_HAND_STAT != 'Horizontally_Slanted_Up':
            left_poses = [[0.6, 0.5, 1.1],
                    [left_target_pose[0], left_target_pose[1]+0.01, left_target_pose[2]+0.18],
                    [left_target_pose[0], left_target_pose[1]+0.04, left_target_pose[2]],
                    [left_target_pose[0]-0.02, left_target_pose[1]-0.06, left_target_pose[2]]]
            for i in range(3):
                left.set_pose_target(NicolPose(left_poses[i], ORIENTATION_DICT_LEFT['Horizontally_Slanted_Up']))
            LEFT_EE_POSE = left.get_eef_pose().position.as_list()
            LEFT_EE_POSE = [LEFT_EE_POSE[0]-0.03, LEFT_EE_POSE[1]-0.07, LEFT_EE_POSE[2]+0.01]
            left.set_pose_target(NicolPose(LEFT_EE_POSE, ORIENTATION_DICT_LEFT['Horizontally_Slanted_Up']))
            LEFT_HAND_STAT = 'Horizontally_Slanted_Up'
            LEFT_FINGER_STAT = 'Hold_Up'
            left.set_joint_position_for_hand(quat)
        # CONTROLLER_attach = left_attachpoint
        # CONTROLLER_sensor = left_sensor
    else:
        detected = sim.checkProximitySensor(right_sensor, sim.handle_all)[0]
        right_target_pose = sim.getObjectPosition(sim.getObject('/'+obj_name+'_right'), sim.handle_world)
        if 'cup' in obj_name:
            return "hold_up is failed, as the chosed object is not suitable to hold up." + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
        if obj_pose[1]>0.25:
            return "hold_up is failed, out of its area." + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
        if RIGHT_FINGER_STAT == 'Closed' and detected:
            return "the right hand is already occupied for grasping and holding some object." + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
        else:
            release('right')
        if RIGHT_HAND_STAT != 'Horizontally_Slanted_Up':
            right_poses = [[0.6, -0.5, 1.1],
                    [right_target_pose[0], right_target_pose[1]-0.01, right_target_pose[2]+0.18],
                    [right_target_pose[0], right_target_pose[1]-0.04, right_target_pose[2]],
                    [right_target_pose[0]-0.02, right_target_pose[1]+0.06, right_target_pose[2]]]
            for i in range(3):
                right.set_pose_target(NicolPose(right_poses[i], ORIENTATION_DICT_RIGHT['Horizontally_Slanted_Up']))
            RIGHT_EE_POSE = right.get_eef_pose().position.as_list()
            RIGHT_EE_POSE = [RIGHT_EE_POSE[0]-0.03, RIGHT_EE_POSE[1]+0.06, RIGHT_EE_POSE[2]+0.01]
            right.set_pose_target(NicolPose(RIGHT_EE_POSE, ORIENTATION_DICT_RIGHT['Horizontally_Slanted_Up']))
            right.set_joint_position_for_hand(quat)
            RIGHT_HAND_STAT = 'Horizontally_Slanted_Up'
            RIGHT_FINGER_STAT = 'Hold_Up'
    left_detected = sim.checkProximitySensor(left_sensor, sim.handle_all)[0]
    right_detected = sim.checkProximitySensor(right_sensor, sim.handle_all)[0]
    # attach the graspable object
    if left_detected and side == 'left':
        read_result = sim.readProximitySensor(left_sensor)
        detected_handle = read_result[3]
        sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(detected_handle, left_attachpoint, True)
        obj_name_grasped = sim.getObjectAlias(read_result[3], -1)
        if '_respondable' in obj_name_grasped:
            obj_name_grasped = obj_name_grasped.replace('_respondable', '')
        # result = f"The robot's {side} hand is prepared to hold the {obj_name} from the side."
        # result = f"The robot's hands are now holding the {obj_name_grasped} together."
    elif right_detected and side == 'right':
        read_result = sim.readProximitySensor(right_sensor)
        detected_handle = read_result[3]
        sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(detected_handle, right_attachpoint, True)
        obj_name_grasped = sim.getObjectAlias(read_result[3], -1)
        if '_respondable' in obj_name_grasped:
            obj_name_grasped = obj_name_grasped.replace('_respondable', '')
    result = f"The robot's {side} hand is prepared to hold the {obj_name} from the side."
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT

def push_to(side:str, source_obj_name:str, target_obj_name:str):
    global LEFT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_HAND_STAT, RIGHT_FINGER_STAT
    hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT if side == 'left' else ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    if side == 'left' and LEFT_FINGER_STAT == 'Closed' or side == 'right' and RIGHT_FINGER_STAT == 'Closed':
        return f"The robot's {side} hand is already occupied for grasping some object." + hand_finger_info
    if 'Origin' in source_obj_name:
        return f"You can not push such object." + hand_finger_info
    try:
        source_obj = sim.getObject('/'+source_obj_name+'_respondable')
    except Exception as e:
        try:
            source_obj = sim.getObject('/'+source_obj_name)
        except Exception as e:
            return f"The object {source_obj_name} is not in the scene." + hand_finger_info
    source_pose = np.array(sim.getObjectPosition(source_obj, sim.handle_world))
    sim.setObjectInt32Parameter(source_obj, sim.shapeintparam_static, 1)
    try:
        target_pose = np.array(sim.getObjectPosition(sim.getObject('/'+target_obj_name), sim.handle_world))
    except Exception as e:
        return f"The object {target_obj_name} is not in the scene." + hand_finger_info
    if (target_obj_name != "overlap_area" and target_pose[2] - source_pose[2] > 0.2) or target_obj_name == "serve_point":
        return f"You can not push the object to {target_obj_name}." + hand_finger_info
    if side == 'left':
        release('left')
        delta_move('left', 'left', 0.06)
        LEFT_HAND_STAT = 'Vertical'
        result = move_single_to_pose('left', source_obj_name, 'Vertical')
        if 'failed' in result: return result + hand_finger_info
        LEFT_HAND_STAT = 'Vertical'
        if sim.checkProximitySensor(left_sensor, sim.handle_all)[0]:
            read_result = sim.readProximitySensor(left_sensor)
            detected_handle = read_result[3]
            sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
            sim.setObjectParent(detected_handle, left_attachpoint, True)
        distance, i = 0.5, 0
        while distance > 0.12:
            i+=1
            source_pose = np.array(sim.getObjectPosition(sim.getObject('/'+source_obj_name), sim.handle_world))
            target_pose = np.array(sim.getObjectPosition(sim.getObject('/'+target_obj_name), sim.handle_world))
            distance = dist.euclidean(source_pose, target_pose)
            # distance = abs(target_pose[1] - source_pose[1])
            if i<6:
                delta_move(side, 'right', 0.06)
            else:
                break
        # if source_obj_name != 'Bowl': move_single_to_pose('left', target_obj_name, 'Vertical', 'left')
        sim.setObjectParent(detected_handle, -1, True)
        delta_move(side, 'left', 0.06)
        sim.setObjectOrientation(source_obj, sim.handle_world, [-0.00010142725493734526, -0.00012479866544761421, 1.6243816263293211])
        return f"The robot's left hand has pushed the {source_obj_name} to the {target_obj_name}. LEFT_HAND: {LEFT_HAND_STAT} LEFT_FINGER: {LEFT_FINGER_STAT}"
    else:
        release('right')
        delta_move('right', 'right', 0.06)
        RIGHT_HAND_STAT = 'Vertical'
        result = move_single_to_pose('right', source_obj_name, 'Vertical')
        if 'failed' in result: return result + hand_finger_info
        RIGHT_HAND_STAT = 'Vertical'
        if sim.checkProximitySensor(right_sensor, sim.handle_all)[0]:
            read_result = sim.readProximitySensor(right_sensor)
            detected_handle = read_result[3]
            sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
            sim.setObjectParent(detected_handle, right_attachpoint, True)
        distance, i = 0.5, 0
        while distance > 0.12:
            i+=1
            source_pose = np.array(sim.getObjectPosition(sim.getObject('/'+source_obj_name), sim.handle_world))
            target_pose = np.array(sim.getObjectPosition(sim.getObject('/'+target_obj_name), sim.handle_world))
            distance = dist.euclidean(source_pose, target_pose)
            # delta_move(side, 'left', 0.06)
            if i<6:
                delta_move(side, 'left', 0.06)
            else:
                break
        # if source_obj_name != 'Bowl': move_single_to_pose('left', target_obj_name, 'Vertical', 'right')
        sim.setObjectParent(detected_handle, -1, True)
        delta_move(side, 'right', 0.06)
        if source_obj_name == "Bowl":
            sim.setObjectOrientation(source_obj, sim.handle_world, [-0.00010142725493734526, -0.00012479866544761421, 1.6243816263293211])
        return f"The robot's right hand has pushed the {source_obj_name} to the {target_obj_name}. RIGHT_HAND: {LEFT_HAND_STAT} RIGHT_FINGER: {LEFT_FINGER_STAT}"

def main(args):
    command = args.command
    para = json.loads(args.para)
    global LEFT_HAND_STAT, RIGHT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_FINGER_STAT
    if args.side == 'left': LEFT_HAND_STAT, LEFT_FINGER_STAT = args.hand_state, args.finger_state
    elif args.side == 'right': RIGHT_HAND_STAT, RIGHT_FINGER_STAT = args.hand_state, args.finger_state
    if command == 'move_and_grasp': 
        if para['obj_name'] == 'Apple' or para['obj_name'] == 'Banana':
            return top_grasp(args.side, **para)
        else:
            return side_grasp(args.side, **para)
    elif command == 'release': 
        controller_ori = LEFT_HAND_STAT if args.side == 'left' else RIGHT_HAND_STAT
        controller_sensor = left_sensor if args.side == 'left' else right_sensor
        detected = sim.checkProximitySensor(controller_sensor, sim.handle_all)[0]
        if detected:
            read_result = sim.readProximitySensor(controller_sensor)
            detected_handle = read_result[3]
            sim.setObjectParent(detected_handle, -1, True)
            if controller_ori == 'Horizontally_Down':
                sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 0)
            # sim.step()
        return release(args.side)
    elif command == 'move_above': 
        if para['obj_name'] == "serve_point":
            return f'You can not move above the serve point!'
        else:
            return move_single_to_pose(args.side, para['obj_name'], off_set = 'up')
    elif command == 'pour_out': 
        try:
            ball = sim.getObject("/big_ball")
            sim.setObjectInt32Parameter(ball, sim.shapeintparam_static, 0)
        except Exception as e:
            return 'There is no water inside the container!'
        return flip_down(args.side, **para)
    elif command == 'hold_up': return hold_up_single(args.side, **para)
    elif command == 'move_to': 
        return move_single_to_pose(args.side, **para)
    elif command in ('wait', 'support_grasped'): 
        side = args.side
        if side == 'left':
            hand_stat = f'LEFT_HAND: {LEFT_HAND_STAT}'
            finger_stat = f'LEFT_FINGER: {LEFT_FINGER_STAT}'
        else:
            hand_stat = f'RIGHT_HAND: {RIGHT_HAND_STAT}'
            finger_stat = f'RIGHT_FINGER: {RIGHT_FINGER_STAT}'
        return f"The robot's {side} hand is keeping the current status. {hand_stat} {finger_stat}" 
    elif command == 'reset': return reset(args.side)
    elif command == 'push_to': return push_to(args.side, **para)
    else: return f'You chosed the unknown command for the chosed hand.'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subprocess Controller Parameters")
    parser.add_argument('--side', type=str, help="Which side of the hand that executes the command")
    parser.add_argument('--command', type=str, help="The command for the side controller")
    parser.add_argument('--para', type=str, help="The parameters for the command")
    parser.add_argument('--hand_state', type=str, help="The hand state for the side controller")
    parser.add_argument('--finger_state', type=str, help="The finger state for the side controller")
    args = parser.parse_args()
    result = main(args)
    if args.side == 'left':
        result = 'LEFT_RESULT: ' + result
    else:
        result = 'RIGHT_RESULT: ' + result
    print(result)
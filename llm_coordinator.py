#!/usr/bin/env python
"""
LLM Tools design for LABOR Agent on NICOL Bimanual Robot
Author: Kun Chu (kun.chu@uni-hamburg.de)
Copyright 2024, Planet Earth
"""

from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict
import time
import numpy as np
import re
import subprocess
import json

# #### The main LLM-based controller
from nicol_api.nicol_env import NicolFactory
from nicol_api.base import NicolPose
from coppeliasim_zmqremoteapi_client import *


############################ Initialize ######################################
nicol = NicolFactory().create_nicol('coppelia', scene="./nicol.ttt", start_scene=False, talker=False)
head  = nicol.head()    # 3D-Printed head structure
left  = nicol.left()    # Left OpenManipulator + RH8D
right = nicol.right()   # Right OpenManipulator + RH8D
sim = nicol.nicol_adapter.sim
left_sensor = sim.getObject('/l_sensor')
left_attachpoint = sim.getObject('/l_palm_attachPoint')
right_sensor = sim.getObject('/r_sensor')
right_attachpoint = sim.getObject('/r_palm_attachPoint')
print("\n"*30, "#"*114, '\n', "#"*114, '\n', 'The simulation starts!')
RIGHT_FINGER_STAT, RIGHT_HAND_STAT = 'Open', 'Vertical'
LEFT_FINGER_STAT, LEFT_HAND_STAT = 'Open', 'Vertical'

import math
def reset_global():
    global RIGHT_FINGER_STAT
    global LEFT_FINGER_STAT
    RIGHT_FINGER_STAT = LEFT_FINGER_STAT = 'Open'
    global LEFT_HAND_STAT
    global RIGHT_HAND_STAT
    LEFT_HAND_STAT = RIGHT_HAND_STAT = 'Vertical'

def move_both_to_poses(obj_name:str):
    handle = sim.getObject('/'+obj_name)
    new_pose = sim.getObjectPosition(handle, sim.handle_world)
    left_pose = left.get_eef_pose().position.as_list()
    right_pose = right.get_eef_pose().position.as_list()
    mid_point = [(left_pose[i]+right_pose[i])/2 for i in range(0,3)]
    left_pose_target = [(left_pose[i]+ new_pose[i]-mid_point[i]) for i in range(0, 3)]
    right_pose_target = [(right_pose[i]+ new_pose[i]-mid_point[i]) for i in range(0, 3)]
    left_ori = left.get_eef_pose().orientation
    left_ori = [left_ori.x, left_ori.y, left_ori.z, left_ori.w]
    right_ori = right.get_eef_pose().orientation
    right_ori = [right_ori.x, right_ori.y, right_ori.z, right_ori.w]
    steps = 0
    right_step_pose = left_pose
    left_step_pose = left_pose
    step_length = 0.065
    steps = 0
    for i in range(0, 3):
        if math.floor((left_pose_target[i]-left_pose[i])/step_length) > steps:
            steps = math.floor((left_pose_target[i]-left_pose[i])/step_length)
    head.set_pose_target(NicolPose([new_pose[0], new_pose[1], new_pose[2]+0.5], [0,0,0,0]))
    for i in range(steps):
        right_step_pose = [min(right_pose_target[0], right_pose[0]+i*step_length), min(right_pose_target[1], right_pose[1]+i*step_length), min(right_pose_target[2], right_pose[2]+i*step_length)]
        left_step_pose = [min(left_pose_target[0], left_pose[0]+i*step_length), min(left_pose_target[1], left_pose[1]+i*step_length), min(left_pose_target[2], left_pose[2]+i*step_length)]
        nicol.set_pose_target_for_both_arms(NicolPose(left_step_pose, left_ori), NicolPose(right_step_pose,right_ori))
        nicol.set_pose_target_for_both_arms(NicolPose(left_step_pose, left_ori), NicolPose(right_step_pose,right_ori))
    nicol.set_pose_target_for_both_arms(NicolPose(left_pose_target, left_ori), NicolPose(right_pose_target,right_ori))
    nicol.set_pose_target_for_both_arms(NicolPose(left_pose_target, left_ori), NicolPose(right_pose_target,right_ori))
    result = f"The NICOL robot's both hands have moved simultaneously to the new positon where the {obj_name} is at the middle point of two hands. "
    return result

# Filter results from the stdouts
##############################################################################
def extract_parts(side, text):
    # Pattern to match the sentence
    sentence_pattern = r"RIGHT_RESULT: (.*?)\. RIGHT_HAND:" if side == "right" else r"LEFT_RESULT: (.*?)\. LEFT_HAND:"
    # Pattern to match the RIGHT_HAND value
    hand_pattern = r"RIGHT_HAND: (\w+)" if side == "right" else r"LEFT_HAND: (\w+)"
    # Pattern to match the RIGHT_FINGER value
    finger_pattern = r"RIGHT_FINGER: (\w+)" if side == "right" else r"LEFT_FINGER: (\w+)"

    # Extract the sentence
    sentence_match = re.search(sentence_pattern, text)
    sentence = sentence_match.group(1) if sentence_match else None

    # Extract the RIGHT_HAND value
    hand_match = re.search(hand_pattern, text)
    hand = hand_match.group(1) if hand_match else None

    # Extract the RIGHT_FINGER value
    finger_match = re.search(finger_pattern, text)
    finger = finger_match.group(1) if finger_match else None

    return sentence, hand, finger
##############################################################################

# Get Arm Pose
##############################################################################
class GetArmStateInput(BaseModel):
    """input for get_both_arm_poses"""
    empty: Dict = Field(description="empty dict", examples=[{}])
        
class GetArmStateTool(BaseTool):
    name = "get_both_arm_poses"
    description = """
        Useful for when you want to know the current pose of the robot. No parameter needed from input.
        Outputs a dictionary containing: 
        the left hand's coordinate in the global coordinate system ['left_palm'], 
        the right hand's coordinate in the global coordinate system ['right_palm'], 
        the left hand's orientation in words ['left_hand_orientation'], 
        the right hand's orientation in words ['right_hand_orientation'], 
        the left hand's finger state ['left_finger_status'], 
        and the right hand's finger state ['right_finger_status']"""
    # args_schema: Type[BaseModel] = GetArmStateInput
    def _run(self, empty=None):
        left_palm_pos = left.get_eef_pose().position.as_list()
        right_palm_pos = right.get_eef_pose().position.as_list()
        result_dict = {
            'left_hand_pose': [round(left_palm_pos[0],3), round(left_palm_pos[1],3), round(left_palm_pos[2]-0.86,3)],
            'right_hand_pose': [round(right_palm_pos[0],3), round(right_palm_pos[1],3), round(left_palm_pos[2]-0.86,3)],
            'left_hand_orientation': LEFT_HAND_STAT,
            'right_hand_orientation': RIGHT_HAND_STAT,
            'left_finger_status': LEFT_FINGER_STAT,
            'right_finger_status': RIGHT_FINGER_STAT
            }
        return result_dict
##############################################################################

# Get object position
#############################################################################
class GetObjPosInput(BaseModel):
    """Inputs for get_obj_position"""
    obj_name: str = Field(description="One single object name", examples='obj')

class GetObjPosTool(BaseTool):
    name = "get_object_position"
    description = """
        Useful when you want to know the specified object's position."""
    args_schema: Type[BaseModel] = GetObjPosInput
    def _run(self, obj_name):
        try:
            obj_handle = sim.getObject('/'+obj_name)
            # objA_handle = sim.getObject('/'+obj_A)
            obj_pose = np.array(sim.getObjectPosition(obj_handle, sim.handle_world))
            return [round(pos, 2) for pos in obj_pose]
        except Exception as e:
            return 'There is no such object in the current environment!'
#############################################################################


##############################################################################
class LABORControlInput(BaseModel):
    """Inputs for labor_control"""
    param_des: str = """
Description of parameter inputs for each command
- 'obj_name': some points names from the task description.
"""
    left_command: str = Field(description=
        """Command chosed for the LEFT hand. 
        Avaliable choices for 'left_command' are chosed from the following commands set: [move_to, move_and_grasp, move_above, push_to, pour_out, hold_up, release, reset, wait]. """)
    left_para: Dict[str, str] = Field(description=
        param_des+ """
        Parameters of commands chosed for the LEFT hand in terms of the Dict. 
        parameters of move_to: 'obj_name'
        parameters of move_and_grasp: 'obj_name'
        parameters of move_above: 'obj_name'
        parameters of hold_up: 'obj_name'
        parameters of push_to: 'source_obj_name', 'target_obj_name'
        parameters of pour_out: {}
        parameters of release: {}
        parameters of wait: {}
        parameters of reset: {}
        Generate the list of required parameters with appropriate fields (Dict).""")
    right_command: str = Field(description=
        """Command chosed for the RIGHT hand. 
        Avaliable choices for 'left_command' are chosed from the following commands set: [move_to, move_and_grasp, move_above, push_to, pour_out, hold_up, release, reset, wait]. """)
    right_para: Dict[str, str] = Field(description=
        param_des+ """
        Parameters of commands chosed for the RIGHT hand in terms of the Dict. 
        parameters of move_to: 'obj_name'
        parameters of move_and_grasp: 'obj_name'
        parameters of move_above: 'obj_name'
        parameters of hold_up: 'obj_name'
        parameters of push_to: 'source_obj_name', 'target_obj_name'
        parameters of pour_out: {}
        parameters of release: {}
        parameters of wait: {}
        parameters of reset: {}
        Generate the list of required parameters with appropriate fields (Dict).""")

##############################################################################
global LEFT_COMMANDS, RIGHT_COMMANDS, LEFT_PARA, RIGHT_PARA, TASK_SUCCESS, LEFT_ACTION_FEEDBACK, RIGHT_ACTION_FEEDBACK
LEFT_COMMANDS = []
LEFT_PARA = []
RIGHT_COMMANDS = []
RIGHT_PARA = []
LEFT_ACTION_FEEDBACK = []
RIGHT_ACTION_FEEDBACK = []

##############################################################################
class LABORControlTool(BaseTool):
    name = "labor_control"
    description = """
Applicable robot skills to control the hand are: 
- move_to: to move one hand to the target position.
- push_to: to control one hand to push some large object horizontally to the target position.
- move_and_grasp: to move one hand to the target object position and grasp it, the hand will hold the targeted object if no release executed. You can only grasp one object with one hand at a time.
- move_above: to move one hand above the target position (necessary before droping objects or pouring out water)
- hold_up: to move one hand to hold up a large object from the bottom side. Useful for holding up some large object together with another hand.
- release: to open one hand to release grasped objects. Do not release objects in the overlap area.
- pour_out: to turn one wrist to flip down grasped object to pour its content out (must above some container)
- reset: to reset one hand to its original status
- wait: the hand, including any possible grasped objects, holds on its present states
"""
    args_schema: Type[BaseModel] = LABORControlInput
    def _run(self, left_command, left_para, right_command, right_para):
        head.set_pose_target(NicolPose([0.8, 0.0, 1], [0, 0, 0, 0]))
        global LEFT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_HAND_STAT, RIGHT_FINGER_STAT, LEFT_COMMANDS, LEFT_PARA, RIGHT_COMMANDS, RIGHT_PARA, LEFT_ACTION_FEEDBACK, RIGHT_ACTION_FEEDBACK
        LEFT_COMMANDS.append(left_command)
        LEFT_PARA.append(left_para)
        RIGHT_COMMANDS.append(right_command)
        RIGHT_PARA.append(right_para)
        if (left_command == right_command == 'move_to' and left_para == right_para and left_para['obj_name'] == 'serve_point'):
            if LEFT_HAND_STAT == RIGHT_HAND_STAT == 'Horizontally_Slanted_Up':
                left_result = right_result = move_both_to_poses(left_para['obj_name'])
            else:
                left_result = right_result = []
        else:
            left_proc = subprocess.Popen(['python', 'nicol_controller.py', "--side=left", f"--command={left_command}", f"--para={json.dumps(left_para)}", f"--hand_state={LEFT_HAND_STAT}", f"--finger_state={LEFT_FINGER_STAT}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(1)
            right_proc = subprocess.Popen(['python', 'nicol_controller.py', "--side=right", f"--command={right_command}", f"--para={json.dumps(right_para)}", f"--hand_state={RIGHT_HAND_STAT}", f"--finger_state={RIGHT_FINGER_STAT}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            l_stdout, _ = left_proc.communicate()
            r_stdout, _ = right_proc.communicate()
            left_result, left_hand, left_finger = extract_parts("left", l_stdout)
            right_result, right_hand, right_finger = extract_parts("right", r_stdout)
            if left_result is None:
                print(l_stdout)
            if right_result is None:
                print(r_stdout)
            LEFT_HAND_STAT, LEFT_FINGER_STAT = left_hand, left_finger
            RIGHT_HAND_STAT, RIGHT_FINGER_STAT = right_hand, right_finger
        # print('left_result: ', left_result)
        # print('right_result: ', right_result)
        LEFT_ACTION_FEEDBACK.append(left_result)
        RIGHT_ACTION_FEEDBACK.append(right_result)
        return left_result, right_result
##############################################################################

system_prompt = """ \n
You are a humanoid robot on a worktable to solve the task given by a human user in a simulated environment.

## Background: 
Coordinates are provided in terms of [x, y, z] in meters in 3D world. The x-coordinate is from back to front, y-coordiConcept atomic is a, promonate is from right to left, and z-coordinate is from bottom to up.

- There are three areas on the worktable:
- Left Area (y coordinate y>0.2): only left hand can reach;
- Right Area (y coordinate y<-0.2): only right hand can reach;
- Overlap Area (y coordinate -0.2<y<0.2): at the middle side of the table, both hands can reach.
- The robot's both hand are initialized vertical and open at beginning. \n"""

guided_prompt = """
##Guidances:
Firstly, you should analysize the areas that the task-related objects are located based on the coordinates. Then, you should follow the rules as below:

If task-related objects are at different areas (uncoordination stages, two hands required)
1) *uncoord_both*: two hands both manipulate the object in their own area independently in parallel, like grasping or releasing or lifting different objects can be executed seperately at the same time.

If task-related objects are at the same area (uncoordination stages, ONLY one single hand required)
2) *unimanual_left*: only left hand acts in sequence in left area to accomplish the task, right hand is not needed. Spatial pre-conditions need to be considered for left hand's sequential manipulation.
3) *unimanual_right*: only right hand acts in sequence in right area to accomplish the task, left hand is not needed. Spatial pre-conditions need to be considered for right hand's sequential manipulation.

If the task accomplishment requires the object in another area, then it is necessary to move objects to the overlap area, like pushing bowl or lifting cups, and then control two hands in cooperation.

Cooperation between two hands in the overlap area (coordination stages, two hands required)
4) *coord_both*: two hands manipulate the object in the overlap area dependently. 
a. *sync*: both hands act symmetrically and simultaneously for mutual supportive collaboration, e.g., jointly lifting some large object together. left and right hand's action needs to be the same and executed at the same time.
b. *async_left*: the right hand's action relys on the left hand's action first to meet spatial pre-conditions.
c. *async_right*: the left hand's action relys on the right hand's action first to meet spatial pre-conditions.

Spatial temporal pre-conditions need to be strictly met at all stages, e.g. pouring out water or releasing objects must be done after moving above the receiving object.

You must decompose the task into an updated list of appropriate stages above, and then generate an appropriate action plan for the next step. Update the list based on the feedback from the environment. Make the plan short with least action steps.
"""

class GPT_Controller():
    def __init__(self, model_name, use_labor=False):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.use_labor = use_labor
        self.tools = [LABORControlTool(), GetArmStateTool(), GetObjPosTool()]
        self.agent = initialize_agent(
            tools = self.tools, 
            llm = self.llm, 
            agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
            verbose = True)
        self.records = {'left_command':LEFT_COMMANDS, 
                        'left_para':LEFT_PARA, 
                        'right_command':RIGHT_COMMANDS, 
                        'right_para':RIGHT_PARA, 
                        'left_feedback':LEFT_ACTION_FEEDBACK, 'right_feedback':RIGHT_ACTION_FEEDBACK}
        self.guided_prompt = guided_prompt

    def run(self, task):
        if self.use_labor:
            self.user_input = system_prompt + task.task_des + self.guided_prompt
        else:
            self.user_input = system_prompt + task.task_des
        self.agent.invoke(self.user_input)
        self.records = {'left_command':LEFT_COMMANDS, 
                        'left_para':LEFT_PARA, 
                        'right_command':RIGHT_COMMANDS, 
                        'right_para':RIGHT_PARA, 
                        'left_feedback':LEFT_ACTION_FEEDBACK, 'right_feedback':RIGHT_ACTION_FEEDBACK}
    def reset(self):
        global RIGHT_COMMANDS, LEFT_COMMANDS, RIGHT_PARA, LEFT_PARA, LEFT_ACTION_FEEDBACK, RIGHT_ACTION_FEEDBACK
        RIGHT_COMMANDS = []
        LEFT_COMMANDS = []
        RIGHT_PARA = []
        LEFT_PARA = []
        LEFT_ACTION_FEEDBACK = []
        RIGHT_ACTION_FEEDBACK = []
        self.records = {'left_command':LEFT_COMMANDS, 'left_para':LEFT_PARA, 'right_command':RIGHT_COMMANDS, 'right_para':RIGHT_PARA, 'left_feedback':LEFT_ACTION_FEEDBACK, 'right_feedback':RIGHT_ACTION_FEEDBACK}
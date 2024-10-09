#!/usr/bin/env python
"""
Task design for NICOL Bimanual Robot
Tasks: ServeWater, ControlScissors, HoldBowl
Author: Kun Chu (kun.chu@uni-hamburg.de)
Copyright 2024, Planet Earth
"""
import random
import os
from llm_coordinator import *
path = os.path.dirname(os.path.abspath(__file__))
random.seed(1234)

"""
Task design structure:
task -- load model
task -- add description
task -- check success condition
task -- self run (hard-coded skill chains)
"""

#########################################################################################
class ServeWaterTask():
    def __init__(self) -> None:
        try: sim.removeModel(sim.getObject('/Bowl_Apple_Banana'))
        except Exception as e: pass
        try:
            self.blue_cup = sim.getObject("/Origin_blue_cup")
            self.yellow_cup = sim.getObject("/Origin_yellow_cup")
            self.ball = sim.getObject("/big_ball")
            self.sensor_1 = sim.getObject("/yellow_cup_sensor")
            self.sensor_2 = sim.getObject("/serve_point_sensor")
        except Exception as e:
            self.model_handle = sim.loadModel(path + '/task_ttms/ServeWater.ttm')
            self.blue_cup = sim.getObject("/Origin_blue_cup")
            self.yellow_cup = sim.getObject("/Origin_yellow_cup")
            self.ball = sim.getObject("/big_ball")
            self.sensor_1 = sim.getObject("/yellow_cup_sensor")
            self.sensor_2 = sim.getObject("/serve_point_sensor")
        self.name = "ServeWater"
        sim.setObjectInt32Parameter(self.ball, sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/blue_cup_respondable"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/yellow_cup_respondable"), sim.shapeintparam_static, 1)
        self.reset(index=0, task_type = 'left_blue_right_yellow')
     
    def reset(self, index=None, task_type=None):
        # reset_global()
        z1 = z2 = 0.86
        x1 = x2 = y1 = y2 = 0
        # x1, y1, x2, y2 = 0.52, 0.43, 0.44, -0.33
        if task_type == 'both_right':
            while abs(y1)-abs(y2) < 0.15:
                x1, y1 = round(random.uniform(0.3, 0.45), 2), round(random.uniform(-0.45, -0.3),2)
                x2, y2 = min(0.6, x1+0.2), round(random.uniform(-0.45, -0.2),2)
            self.blue_spatial_relation = 'in the right area'
            self.yellow_spatial_relation = 'in the right area'
        elif task_type == 'both_left':
            while abs(y1)-abs(y2) < 0.15:
                x1, y1 = round(random.uniform(0.4, 0.45), 2), round(random.uniform(0.3, 0.45),2)
                x2, y2 = min(0.6, x1+0.2), round(random.uniform(0.2, 0.45),2)
            self.blue_spatial_relation = 'in the left area'
            self.yellow_spatial_relation = 'in the left area'
        elif task_type == 'left_blue_right_yellow':
            while abs(y1)-abs(y2) < 0.15:
                x1, y1 = round(random.uniform(0.4, 0.45), 2), round(random.uniform(0.3, 0.45),2)
                x2, y2 = min(0.6, x1+0.2), round(random.uniform(-0.45, -0.2),2)
            self.blue_spatial_relation = 'in the left area'
            self.yellow_spatial_relation = 'in the right area'
        elif task_type == 'left_yellow_right_blue':
            while abs(y1)-abs(y2) < 0.15:
                x1, y1 = round(random.uniform(0.3, 0.45), 2), round(random.uniform(-0.45, -0.3),2)
                x2, y2 = min(0.6, x1+0.2), round(random.uniform(0.2, 0.45),2)
            self.blue_spatial_relation = 'in the right area'
            self.yellow_spatial_relation = 'in the left area'
        sim.setObjectPose(self.blue_cup, sim.handle_world, [x1, y1, z1, 0,0,0,1])
        sim.setObjectPose(self.yellow_cup, sim.handle_world, [x2, y2, z2, 0,0,0,1])
        sim.setObjectPose(self.ball, sim.handle_world, [x1, y1, z1+0.01, 0,0,0,1])
        sim.setObjectInt32Parameter(self.ball, sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/blue_cup_respondable"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/yellow_cup_respondable"), sim.shapeintparam_static, 1)
        self.blue_cup_pose =  [x1, y1, z1]
        self.yellow_cup_pose = [x2, y2, z2]
        self.short_des = f"""blue_cup {[x1, y1, z1]} with water, yellow cup {[x2, y2, z2]} without water."""
        self.task_des = f"""
##Environment Setting: 
There are two cups placed on the table:
- One blue cup is filled with water, with the coordinate as {[x1, y1, z1]}
- One yellow cup is empty, with the coordinate as {[x2, y2, z2]}

- Marked points names are listed as: ['Origin_blue_cup', 'Origin_yellow_cup', 'yellow_cup', 'blue_cup', 'serve_point', 'overlap_area'], in which the name entity starts with "Origin_" indicates its initial position (not used for grasping), other names without this prefix are the current/dynamic positions during the task.
- serve point is at [0.8, 0.0, 1.2], where the human user is waiting for the water.

##Task:
Pour the water from the blue cup into the yellow cup, put the blue cup back, and serve the water to the human user at the serve point. Do not release the cup at the serving point.

"""

    def check_success(self):
        cup_respond = sim.getObject("/yellow_cup_respondable")
        detect_result_1 = sim.checkProximitySensor(self.sensor_1, self.ball)
        detect_result_2 = sim.checkProximitySensor(self.sensor_2, cup_respond)
        if detect_result_1[0] and detect_result_2[0]:
            return True
        else:
            return False
    
    def self_run(self):
        llm_coordinate = LABORControlTool()
        if self.blue_cup_pose[1] <0 and self.yellow_cup_pose[1] <0:
            llm_coordinate._run('wait', {}, 'move_and_grasp', {'obj_name':'blue_cup'})
            llm_coordinate._run('wait', {}, 'move_above', {'obj_name':'yellow_cup'})
            llm_coordinate._run('wait', {}, 'pour_out', {})
            llm_coordinate._run('wait', {}, 'move_to', {'obj_name': 'Origin_blue_cup'})
            llm_coordinate._run('wait', {}, 'release', {})
            llm_coordinate._run('wait', {}, 'move_and_grasp', {'obj_name':'yellow_cup'})
            llm_coordinate._run('wait', {}, 'move_to', {'obj_name': 'serve_point'})
        elif self.blue_cup_pose[1] >=0 and self.yellow_cup_pose[1] >=0:
        # both left
            llm_coordinate._run('move_and_grasp', {'obj_name':'blue_cup'}, 'wait', {})
            llm_coordinate._run('move_above', {'obj_name':'yellow_cup'}, 'wait', {})
            llm_coordinate._run('pour_out', {}, 'wait', {})
            llm_coordinate._run('move_to', {'obj_name': 'Origin_blue_cup'}, 'wait', {})
            llm_coordinate._run('release', {}, 'wait', {})
            llm_coordinate._run('move_and_grasp', {'obj_name':'yellow_cup'}, 'wait', {})
            llm_coordinate._run('move_to', {'obj_name': 'serve_point'}, 'wait', {})
        elif self.blue_cup_pose[1] <0 and self.yellow_cup_pose[1] >=0:
            llm_coordinate._run('move_and_grasp', {'obj_name': 'yellow_cup'}, 'move_and_grasp', {'obj_name':'blue_cup'})
            llm_coordinate._run('move_to', {'obj_name': 'overlap_area'}, 'wait', {})
            llm_coordinate._run('wait', {}, 'move_above', {'obj_name':'yellow_cup'})
            llm_coordinate._run('wait', {}, 'pour_out', {}) 
            llm_coordinate._run('move_to', {'obj_name': 'serve_point'}, 'move_to', {'obj_name': 'Origin_blue_cup'})
        elif self.blue_cup_pose[1] >=0 and self.yellow_cup_pose[1] <0:
            llm_coordinate._run('move_and_grasp', {'obj_name':'blue_cup'}, 'move_and_grasp', {'obj_name': 'yellow_cup'})
            llm_coordinate._run('wait', {}, 'move_to', {'obj_name': 'overlap_area'})
            llm_coordinate._run('move_above', {'obj_name':'yellow_cup'}, 'wait', {})
            llm_coordinate._run('pour_out', {}, 'wait', {})
            llm_coordinate._run('move_to', {'obj_name': 'Origin_blue_cup'}, 'move_to', {'obj_name': 'serve_point'})


#########################################################################################
class ServeFruitTask():
    def __init__(self) -> None:
        try: sim.removeModel(sim.getObject('/cups_with_balls'))
        except Exception as e: pass
        try: sim.removeModel(sim.getObject('/scissor'))
        except Exception as e: pass
        try:
            self.bowl_object = sim.getObject("/Origin_bowl")
            self.apple_object = sim.getObject("/Apple")
            self.banana_object = sim.getObject("/Banana")
            self.target_sensor = sim.getObject("/serve_point_sensor")
        except Exception as e:
            bowl_handle = sim.loadModel(path + '/task_ttms/ServeFruit.ttm')
            self.bowl_object = sim.getObject("/Origin_bowl")
            self.apple_object = sim.getObject("/Apple")
            self.banana_object = sim.getObject("/Banana")
            self.target_sensor = sim.getObject("/serve_point_sensor")
        sim.setObjectInt32Parameter(sim.getObject("/Bowl_respondable"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/Apple"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/Banana"), sim.shapeintparam_static, 1)
        reset_global()
        self.name = "ServeFruit"
        self.reset(index=0, task_type = 'same_fruits_same_bowl')

    def reset(self, index=None, task_type=None):
        reset_global()
        sim.setObjectInt32Parameter(sim.getObject("/Bowl_respondable"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(self.apple_object, sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(self.banana_object, sim.shapeintparam_static, 1)
        # sim.setObjectInt32Parameter(self.bowl_object, sim.shapeintparam_static, 1)
        z1, z2, z3 = 0.82, 0.81, 0.81
        x1 = x2 = y1 = y2 = 0
        range1, range2 = (-0.65, -0.2), (0.2, 0.65)
        coordinates1 = (0.25, 0.5, 0.45, 0.6)
        coordinates2 = (0.25, -0.5, 0.45, -0.6)
        coordinates3 = (0.45, -0.6, 0.45, 0.6)
        coordinates4 = (0.45, 0.55, 0.45, -0.6)
        chosen_coordinates_1 = random.choice([coordinates1, coordinates2])
        chosen_coordinates_2 = random.choice([coordinates3, coordinates4])

        if task_type == 'same_fruits_same_bowl':
            x1, y1, x2, y2 = chosen_coordinates_1
            if y1 > 0:
                x3, y3, y3_out = 0.6, 0.3, 0.4
            else:
                x3, y3, y3_out = 0.6, -0.3, -0.4
        elif task_type == 'same_fruits_diff_bowl':
            x1, y1, x2, y2 = chosen_coordinates_1
            if y1 > 0:
                x3, y3, y3_out = 0.6, -0.3, -0.4
            else:
                x3, y3, y3_out = 0.6, 0.3, 0.4
        elif task_type == 'diff_fruit_left_bowl':
            x1, y1, x2, y2 = chosen_coordinates_2
            x3, y3, y3_out = 0.6, 0.3, 0.4
        elif task_type == 'diff_fruit_right_bowl':
            x1, y1, x2, y2 = chosen_coordinates_2
            x3, y3, y3_out = 0.6, -0.3, -0.4
        else:
            print("You choose the wrong task!")
        sim.setObjectPose(self.apple_object, sim.handle_world, [x1, y1, z1, 0, 0, 0, 1])
        sim.setObjectPose(self.banana_object, sim.handle_world, [x2, y2, z2, 0, 0, -4, 1])
        sim.setObjectPose(self.bowl_object, sim.handle_world, [x3, y3, z3, 0, 0, 0, 1])
        # x3, y3, z3 = sim.getObjectPose(self.bowl_object, -1)[0:3]
        self.apple_pose = [x1, y1, z1]
        self.banana_pose = [x2, y2, z2]
        self.bowl_pose = [x3, y3, z3]
        self.short_des = f" apple is at {[x1, y1, z1]}), banana is at {[x2, y2, z2]}, and bowl is at {[x3, y3, z3]}."
        self.task_des = f"""\n
## Environment Setting: 
There are three objects at the table:
- an apple is at {self.apple_pose}
- a banana is at {self.banana_pose}
- a large bowl is at {[x3, y3_out, z3]} (not graspable with one single hand) \n
- serve point is at (0.6, 0.0, 1.0) in the middle air. \n
- Marked points position are listed as: ['Origin_right_hand', 'Origin_left_hand', 'Apple', 'Banana', 'Bowl', 'overlap_area', 'serve_point'], in which the name entity starts with "Origin_" indicates its initial position, other names without this prefix are the current/dynamic positions during the task. 

Task: 
Grasp the fruits and release them to the bowl, and serve the bowl to the human user at serve point. 

Notes:
Do not release the bowl at the serve point. 
Do not release the apple and banana in the overlap area.

"""
    def check_success(self):
        apple_detected = sim.checkProximitySensor(self.target_sensor, self.apple_object)[0]
        banana_detected = sim.checkProximitySensor(self.target_sensor, self.banana_object)[0]
        if apple_detected == 0:
            print('The apple is not lifted with the bowl!')
        if banana_detected == 0:
            print('The banana is not lifted with the bowl!')
        if apple_detected>0 and banana_detected>0:
            return True
        else:
            return False
    
    def self_run(self):
        llm_coordinate = LABORControlTool()
        if self.bowl_pose[1] < 0:
            llm_coordinate._run('wait', {}, 'push_to', {'source_obj_name':'Bowl', 'target_obj_name':'overlap_area'})
        else:
            llm_coordinate._run('push_to', {'source_obj_name':'Bowl', 'target_obj_name':'overlap_area'}, 'wait', {})
        if self.apple_pose[1] * self.banana_pose[1] <0:
            if self.apple_pose[1]<0 and self.banana_pose[1]>0:
                llm_coordinate._run('move_and_grasp', {'obj_name':'Banana'}, 'move_and_grasp', {'obj_name': 'Apple'})
                llm_coordinate._run('wait', {}, 'move_above', {'obj_name':'Bowl'})
                llm_coordinate._run('wait', {}, 'release', {})
                llm_coordinate._run('move_above', {'obj_name':'Bowl'}, 'move_to', {'obj_name':'Origin_right_hand'})
                llm_coordinate._run('release', {}, 'wait', {})
            elif self.apple_pose[1]>0 and self.banana_pose[1]<0:
                llm_coordinate._run('move_and_grasp', {'obj_name':'Apple'}, 'move_and_grasp', {'obj_name': 'Banana'})
                llm_coordinate._run('move_above', {'obj_name':'Bowl', 'off_set':'up'}, 'wait', {})
                llm_coordinate._run('release', {}, 'wait', {})
                llm_coordinate._run('move_to', {'obj_name':'Origin_left_hand'}, 'move_above', {'obj_name':'Bowl'})
                llm_coordinate._run('wait', {}, 'release', {})
        else:
            if self.apple_pose[1]>0:
                llm_coordinate._run('move_and_grasp', {'obj_name':'Apple'}, 'wait', {})
                llm_coordinate._run('move_above', {'obj_name':'Bowl'}, 'wait', {})
                llm_coordinate._run('release', {}, 'wait', {})
                llm_coordinate._run('move_and_grasp', {'obj_name':'Banana'}, 'wait', {})
                llm_coordinate._run('move_above', {'obj_name':'Bowl'}, 'wait', {})
                llm_coordinate._run('release', {}, 'wait', {})
            else:
                llm_coordinate._run('wait', {}, 'move_and_grasp', {'obj_name':'Apple'})
                llm_coordinate._run('wait', {}, 'move_above', {'obj_name':'Bowl'})
                llm_coordinate._run('wait', {}, 'release', {})
                llm_coordinate._run('wait', {}, 'move_and_grasp', {'obj_name':'Banana'})
                llm_coordinate._run('wait', {}, 'move_above', {'obj_name':'Bowl'})
                llm_coordinate._run('wait', {}, 'release', {})
        llm_coordinate._run('hold_up', {'obj_name':'Bowl'}, 'hold_up', {'obj_name':'Bowl'})
        llm_coordinate._run('move_to', {'obj_name':'serve_point'}, 'move_to', {'obj_name':'serve_point'})

def create_task(task_name):
    if task_name == "ServeWater":
        return ServeWaterTask()
    elif task_name == "ServeFruit":
        return ServeFruitTask()
    else:
        print("The task is not defined yet!")
        return False
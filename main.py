#!/usr/bin/env python
from tasks import create_task
from llm_coordinator import *
import sys
import time
import argparse

################################ Logger #############################
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
import os
import pandas as pd

specified_columns = ['task_type', 'task_index', 'success', 'left_command', 'left_para', 'right_command', 'right_para', 'left_feedback', 'right_feedback']
def write_record_line(data, record_file):
    if os.path.exists(record_file):
        df = pd.read_csv(record_file)
        df_udpate = pd.concat([df, pd.DataFrame(data, columns=df.columns)]).reset_index(drop=True)
    else:
        df_udpate = pd.DataFrame(data, columns=specified_columns)
    df_udpate.to_csv(record_file, index=False, header=True)
    return

cup_task_types = ['left_blue_right_yellow', 'left_yellow_right_blue', 'both_left', 'both_right']
bowl_task_types = ['same_fruits_same_bowl', 'same_fruits_diff_bowl', 'diff_fruit_left_bowl', 'diff_fruit_right_bowl']

def reset_task(task_name, task, task_index):
    if task_name == 'ServeWater':
        task.reset(task_type=cup_task_types[task_index])
    elif task_name == 'ServeFruit':
        task.reset(task_type=bowl_task_types[task_index])
    else:
        task.reset()

def main(args):
    task = create_task(args.task_name)
    print(task.task_des)
    llm_controller = GPT_Controller(args.model_name, use_labor=args.use_labor)
    
    if args.use_labor:
        method = 'LABOR'
        print('Guided prompt ', llm_controller.guided_prompt)
    else:
        method = 'Baseline'
    if 'gpt-3.5' in args.model_name:
        file_name = './logs/' + args.task_name + '_gpt-3.5_' + method + '1-' + str(args.num_tasks)
        sys.stdout = Logger(file_name + '.txt', sys.stdout)
    elif 'gpt-4' in args.model_name:
        file_name = './logs/' + args.task_name + '_gpt-4_' + method + '1-' + str(args.num_tasks) 
        sys.stdout = Logger(file_name + '.txt', sys.stdout)
    total_num = 0
    success_num = 0
    print("\n","#" * 114)
    print(f"The task {args.task_name} starts!")
    if 'True' == args.use_labor or args.use_labor == True:
        print(f"The LABOR Agent with {args.model_name} is used for the task.")
    else:
        print(f"The Baseline Agent with {args.model_name} is used for the task.")

    if args.task_name == 'ServeWater': 
        task_types = cup_task_types
        task_var_num = len(cup_task_types)
    elif args.task_name == 'ServeFruit': 
        task_types = bowl_task_types
        task_var_num = len(bowl_task_types)
    else:
        return "The task is not supported yet!"

    for i in range(args.num_tasks):
        for j in range(task_var_num):
            total_num += 1
            reset_task(args.task_name, task, j)
            print("\n", "#" * 20)
            print('Task:  ', total_num, task.short_des)
            try:
                if args.use_llm == True:
                    while llm_controller.records['left_command'] == []:
                        llm_controller.run(task)
                else:
                    task.self_run()
            except Exception as e:
                print("Error: \n", e)
                pass
            
            success = task.check_success()
            reset_global()
            if success:
                success_num += 1
                print("The task is successfully accomplished!")
            else:
                print("Unfortunately, the task is failed...")
            if args.use_llm:
                llm_controller.records['task_index'] = total_num
                llm_controller.records['task_type'] = task_types[j]
                llm_controller.records['success'] = success
                write_record_line(llm_controller.records, file_name + '.csv')
                llm_controller.reset()
            sim.stopSimulation()
            time.sleep(1)
            sim.startSimulation()
    total_success_rate = success_num / (args.num_tasks * task_var_num)
    print(f"The total success rate is {total_success_rate}!")
    time.sleep(2)
    sim.stopSimulation()
    print('The simulation ends!')
    print("#"*114)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Parameters")
    parser.add_argument('--use_labor', action='store_true', default=False, help="Wether to use labor or not")
    parser.add_argument('--use_llm', action='store_true', default=False, help="Wether to use LLM or not")
    parser.add_argument('--task_name', type=str, default="ServeFruit", help="Which task to run")
    parser.add_argument('--num_tasks', type=int, default=10, help="How many tasks to run")
    parser.add_argument('--model_name', type=str, default="gpt-4o", help="Which model to use")
    args = parser.parse_args()
    main(args)
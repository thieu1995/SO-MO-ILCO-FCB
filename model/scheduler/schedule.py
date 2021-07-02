#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:23, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from itertools import chain
from copy import deepcopy


class Schedule:

    def __init__(self, problem):
        """Init the Schedule object
        n_fogs: The number of fog instances
        n_clouds: The number of cloud instances
        n_peers: The number of blockchain nodes
        n_tasks: The number of tasks
        """
        self.clouds = problem["clouds"]
        self.fogs = problem["fogs"]
        self.peers = problem["peers"]
        self.tasks = problem["tasks"]

        self.n_clouds = len(self.clouds)
        self.n_fogs = len(self.fogs)
        self.n_peers = len(self.peers)
        self.n_tasks = len(self.tasks)

        self.schedule_clouds_tasks = {} # key: cloud_id, val: list_task_id []
        self.schedule_fogs_tasks = {}
        self.schedule_peers_tasks = {}

    def get_list_task_handler(self):
        task_handlers = {
            # "task_id": "node_id",
            # "2": 3,     # Now each task will be handled by single node (fog or cloud)
        }
        for fog_id, list_task_id in self.schedule_fogs_tasks.items():
            for task_id in list_task_id:
                if task_id not in task_handlers.keys():
                    task_handlers[task_id] = fog_id
        for cloud_id, list_task_id in self.schedule_clouds_tasks.items():
            for task_id in list_task_id:
                if task_id not in task_handlers.keys():
                    task_handlers[task_id] = cloud_id
        return task_handlers

    def get_list_task_handler_with_order(self):
        task_handlers = {
            # "task_id": {
            #       "node_id": [taskId, taskID,...],     those tasks which will be handled before this task_id
            # },
        }
        for fog_id, list_task_id in self.schedule_fogs_tasks.items():
            for idx, task_id in enumerate(list_task_id):
                if task_id not in task_handlers.keys():
                    list_task_before = []
                    for idx_before in range(0, idx):
                        list_task_before.append(list_task_id[idx_before])
                    task_handlers[task_id] = {
                        fog_id: list_task_before
                    }
                else:
                    continue
        for cloud_id, list_task_id in self.schedule_clouds_tasks.items():
            for idx, task_id in enumerate(list_task_id):
                list_task_before = []
                for idx_before in range(0, idx):
                    list_task_before.append(list_task_id[idx_before])
                task_handlers[task_id] = {
                    cloud_id: list_task_before
                }
        return task_handlers

    def is_valid(self) -> bool:
        """
        Check whether this schedule is valid or not
        :return: bool
        """

        ## 1. Total tasks = Total tasks in fogs + total tasks in clouds
        tasks_temp_cloud = [task_id for task_id in self.schedule_clouds_tasks.values()]
        tasks1 = list(chain(*tasks_temp_cloud))        ## Kinda same as set(list) to remove duplicate element
        tasks_temp_fog = [task_id for task_id in self.schedule_fogs_tasks.values()]
        tasks2 = list(chain(*tasks_temp_fog))
        if (len(set(tasks1)) + len(set(tasks2))) != self.n_tasks:
            return False
        return True

    def __repr__(self):
        return f'Schedule {{\n' \
               f'  clouds: {self.schedule_clouds_tasks!r}\n' \
               f'  fogs: {self.schedule_fogs_tasks!r}\n' \
               f'}}'

    def clone(self):
        return deepcopy(self)

    @property
    def total_time(self) -> int:
        cloud_time = max([len(val) for val in self.schedule_clouds_tasks.values()])
        fog_time = max([len(val) for val in self.schedule_fogs_tasks.values()])
        return max(cloud_time, fog_time)
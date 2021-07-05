#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:43, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from config import *
from model.scheduler.schedule import Schedule
from model.scheduler.formulas import power, latency
from model.scheduler.formulas import cost
from numpy import array, math
from numpy.linalg import norm


class Fitness:

    def __init__(self, problem):
        self.clouds = problem["clouds"]
        self.fogs = problem["fogs"]
        self.peers = problem["peers"]
        self.tasks = problem["tasks"]
        self.list_to_dict({'clouds': self.clouds, 'fogs': self.fogs, 'peers': self.peers, 'tasks': self.tasks})

        self._min_power = 0
        self._min_latency = 0
        self._min_cost = 0

    def list_to_dict(self, *attributes):
        for idx, attrs in enumerate(attributes):
            for key, values in attrs.items():
                obj_list = {}
                for obj in values:
                    obj_list[obj.id] = obj
                setattr(self, key, obj_list)

    def set_min_power(self, value: float):
        self._min_power = value

    def set_min_latency(self, value: float):
        self._min_latency = value

    def set_min_cost(self, value: float):
        self._min_cost = value
        
    def get_punished_cost(self, exceeded_cost):
        exceeded_cost = max(0, exceeded_cost)
        return min(0.1, math.log(1 + exceeded_cost))


    def fitness(self, solution: Schedule) -> float:
        power = self.calc_power_consumption(solution)
        latency = self.calc_latency(solution)
        cost = self.calc_cost(solution)

        # assert self._min_power <= power
        # assert self._min_latency <= latency
        # assert self._min_cost <= cost

        if Config.METRICS == 'power':
            return power
        elif Config.METRICS == 'latency':
            return latency
        elif Config.METRICS == 'cost':
            return cost
        elif Config.METRICS == "weighting":
            w = array(Config.OBJ_WEIGHTING_METRICS)
            v = array([power, latency, cost])
            return sum(w * v)
        elif Config.METRICS == "distancing":
            o = array(Config.OBJ_DISTANCING_METRICS)
            v = array([power, latency, cost])
            return norm(o-v)
        elif Config.METRICS == 'min-max':
            o = array(Config.OBJ_MINMAX_METRICS)
            v = array([power, latency, cost])
            return max((v-o)/o)         # Need to minimize the relative deviation of single objective functions
        elif Config.METRICS == "weighting-min":   # The paper of Thang and Khiem
            w = array(Config.OBJ_WEIGHTING_MIN_METRICS_1)
            o = array(Config.OBJ_WEIGHTING_MIN_METRICS_2)
            v = array([power, latency, cost])
            return sum((w * o) / v)
        elif Config.METRICS == "pareto":
            return array([power, latency, cost])
        else:
            print(f'[ERROR] Metrics {Config.METRICS} is not supported in class FitnessManager')

    def calc_power_consumption(self, schedule: Schedule) -> float:
        po = power.data_forwarding_power(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        po += power.computation_power(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        po += power.storage_power(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        return po / 3600

    def calc_latency(self, schedule: Schedule) -> float:
        la_t = latency.transmission_latency(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        la_p = latency.processing_latency(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        la_max = latency.punishment_latency(self.clouds, self.fogs, self.peers, self.tasks, schedule, la_t, la_p)
        # print(la_max)
        return la_p + la_t + la_max

    def calc_cost(self, schedule: Schedule) -> float:
        
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            for idx, task_id in enumerate(list_task_id):
                task = self.tasks[task_id]
                task.reset()

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            for idx, task_id in enumerate(list_task_id):
                task = self.tasks[task_id]
                task.reset()
                    
        for peer_id, list_task_id in schedule.schedule_peers_tasks.items():
            for idx, task_id in enumerate(list_task_id):
                task = self.tasks[task_id]
                task.reset()
                
        co = cost.data_forwarding_cost(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        co += cost.computation_cost(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        co += cost.storage_cost(self.clouds, self.fogs, self.peers, self.tasks, schedule)
        
        punish_cost = 0
        
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            for idx, task_id in enumerate(list_task_id):
                task = self.tasks[task_id]
                punish_cost += self.get_punished_cost(task.total_cost - task.cost_max)

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            for idx, task_id in enumerate(list_task_id):
                task = self.tasks[task_id]
                punish_cost += self.get_punished_cost(task.total_cost - task.cost_max)
                    
        for peer_id, list_task_id in schedule.schedule_peers_tasks.items():
            for idx, task_id in enumerate(list_task_id):
                task = self.tasks[task_id]
                punish_cost += self.get_punished_cost(task.total_cost - task.cost_max)
        print(punish_cost)        
        
        return co
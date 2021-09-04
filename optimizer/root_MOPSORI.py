#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:35, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from config import Config
from sys import exit
from numpy import inf, zeros, argmin, array, mean
from numpy.random import uniform
from optimizer.root import Root
from utils.schedule_util import matrix_to_schedule
from utils.visual.scatter import visualize_2D, visualize_3D
from uuid import uuid4
from copy import deepcopy
from model.scheduler.fitness import Fitness
import math

class Root3(Root):

    ID_IDX = 0
    ID_POS = 1              # Current position
    ID_FIT = 2              # Current fitness
    ID_VEL = 3              # Current velocity
    ID_LOCAL_POS = 4        # Personal best location
    ID_LOCAL_FIT = 5        # Personal best fitness
    n_props = 6

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        self.n_objs = None
        self.WEIGHT = 1e-6
        self.Fit = Fitness(problem)

    def create_solution(self):
        while True:
            pos = uniform(self.lb, self.ub, self.problem["shape"])
            schedule = matrix_to_schedule(self.problem, pos.astype(int))
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                vel = uniform(self.lb, self.ub, self.problem["shape"])
                break
        idx = uuid4().hex
        return [idx, pos, fitness, vel, [pos], [fitness]]
        # [solution, fit, velocity, local_solution, local_fitness]

    # Function to sort by values
    def sort_by_values(self, front: list, obj_list: array):
        sorted_list = []
        obj_list = deepcopy(obj_list)
        while (len(sorted_list) != len(front)):
            idx_min = argmin(obj_list)
            if idx_min in front:
                sorted_list.append(idx_min)
            obj_list[idx_min] = inf
        return sorted_list

    def dominate(self, id1, id2, obj):
        better = False
        for i in range(self.n_objs):
            if obj[i][id1] > obj[i][id2]:
                return False
            elif obj[i][id1] < obj[i][id2]:
                better = True
        return better

    def is_dominate(self, obj1, obj2):
        for i in range(self.n_objs):
            if obj1[i] > obj2[i]:
                return False
        return True
    
    def is_non_dominated(self, obj1, obj2):
        for i in range(self.n_objs):
            if obj1[i] < obj2[i]:
                return True
        return False
    
    def is_better(self, fit_1, fit_2):
        better = True
        # print(fit_1, fit_2)
        for i in range(self.n_objs):
            if fit_1[i] > fit_2[i]:
                return False
        return better

    def step_decay(self, epoch, init_explore_rate = 0.9):
       init_explore_rate = 0.9
       drop = (1 - 1 / (math.e + 3)) 
       epochs_drop = math.floor(math.sqrt(self.epoch))
       explore_rate = init_explore_rate * math.pow(drop, math.floor((1 + epoch)/epochs_drop))
       return explore_rate
   
    def step_decay2(self, epoch, init_er):
       init_explore_rate = init_er
       drop = (1 - 1 / (math.e + 3)) 
       epochs_drop = math.floor(math.sqrt(self.epoch))
       explore_rate = init_explore_rate * math.pow(drop, math.floor((1 + epoch)/epochs_drop))
       return max(explore_rate, 0.02)
   
 
    # Function to carry out NSGA-II's fast non dominated sort
    def fast_non_dominated_sort(self, pop: dict):
        key_list = list(pop.keys())
        # id_list = {}
        # for i in range(0, self.pop_size):
            # id_list[key_list[i]] = i
            
        front = []
        num_assigned_individuals = 0
        indv_ranks = [0 for _ in range(0, self.pop_size)]
        rank = 1
        mx, mn = [0] * self.n_objs, [0] * self.n_objs
        for j in range(self.n_objs):
            mx[j] = max([pop[key_list[i]][self.ID_FIT][j] for i in range(self.pop_size)])
            mn[j] = min([pop[key_list[i]][self.ID_FIT][j] for i in range(self.pop_size)])
        
        while num_assigned_individuals < self.pop_size:
            cur_front = []
            for i in range(self.pop_size):
                if indv_ranks[i] > 0:
                    continue
                be_dominated = False
                
                j = 0
                while j < len(cur_front):
                    idx_1 = key_list[cur_front[j]]
                    idx_2 = key_list[i]
                    if self.is_dominate(pop[idx_1][self.ID_FIT], pop[idx_2][self.ID_FIT]):
                        be_dominated = True
                        break
                    elif self.is_dominate(pop[idx_2][self.ID_FIT], pop[idx_1][self.ID_FIT]):
                        cur_front[j] = cur_front[-1]
                        cur_front.pop()
                        j -= 1
                    j += 1
                        
                if not be_dominated:
                    cur_front.append(i)
            sorted(cur_front, key=lambda x: 
                sum([(pop[key_list[x]][self.ID_FIT][j] - mn[j]) / (mx[j] - mn[j] + 1e-10)
                     for j in range(self.n_objs)]
                ))
            for i in range(len(cur_front)):
                indv_ranks[ cur_front[i] ] = rank
                rank += 1
            front.append(cur_front)
            num_assigned_individuals += len(cur_front)
        return front, indv_ranks

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        pass

    def train(self):
        time_total = time()
        time_bound_start = time()
        self.check_objective()
        self.check_log()

        pop_temp = [self.create_solution() for _ in range(self.pop_size)]
        self.n_objs = len(pop_temp[0][self.ID_FIT])
        # pop = [item for item in pop_temp]
        # print(self.n_objs)
        pop = {item[self.ID_IDX]: item for item in pop_temp}
        training_info = {"Epoch": [], "FrontSize": [], "Time": []}

        if Config.MODE == 'epoch':
            mo_g_best = []
            g_best_dict = {}
            fronts, rank = self.fast_non_dominated_sort(pop)
            for it in fronts[0]:
                mo_g_best.append(pop[list(pop.keys())[it]])
                
            for epoch in range(self.epoch):
                time_epoch_start = time()
                pop = self.evolve(pop, None, epoch, mo_g_best)
                fronts, rank = self.fast_non_dominated_sort(pop)
                current_best = []
                mo_g_best = []
                for it in fronts[0]:
                    current_best.append(list(pop.values())[it][self.ID_FIT])
                for it in fronts[0]:
                    mo_g_best.append(pop[list(pop.keys())[it]])
                g_best_dict[epoch] = array(current_best)
                time_epoch_end = time() - time_epoch_start
                training_info = self.adding_element_to_dict(training_info, ["Epoch", "FrontSize", "Time"], [epoch+1, len(fronts[0]), time_epoch_end])
                if self.verbose:
                    obj = [zeros(len(pop)) for i in range(self.n_objs)]
                    for idx, item in enumerate(pop.values()):
                        for i in range(self.n_objs):
                            obj[i][idx] = float(item[self.ID_FIT][i])
                    # print(obj)
                    visualize_3D(obj)
                    # for j in  range(len(fronts[0])):
                    #     idx = fronts[0][j]
                    #     for i in range(self.n_objs):
                    #         obj[i][j] = float(pop[list(pop.keys())[idx]][self.ID_FIT][i])
                    # obj = [zeros(len(fronts)) for i in range(self.n_objs)]
                    # visualize_3D(obj, 'red', True)
                    value = [0] * self.n_objs
                    for i in range(self.n_objs):
                        value[i] = mean([list(pop.values())[j][self.ID_FIT][i] for j in fronts[0]])
                    for i in range(len(value)):
                        value[i] = round(value[i], 3)
                    print(f'Epoch: {epoch+1}, Front size: {len(fronts[0])}, including {value}, time: {time_epoch_end:.2f} seconds')
                if Config.TIME_BOUND_KEY:
                    if time() - time_bound_start >= Config.TIME_BOUND_VALUE_PER_TASK * self.problem["n_tasks"]:
                        print('====== Over time for training ======')
                        break
            solutions = {}
            g_best = []
            for it in fronts[0]:
                idx = list(pop.keys())[it]
                solutions[idx] = pop[idx]
                g_best.append(pop[idx][self.ID_FIT])
            time_total = time() - time_total
            training_info = self.adding_element_to_dict(training_info, ["Epoch", "FrontSize", "Time"], [epoch + 1, len(fronts[0]), time_total])
            return solutions, array(g_best), g_best_dict, training_info


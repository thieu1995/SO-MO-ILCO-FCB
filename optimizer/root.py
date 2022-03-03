#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:35, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from math import gamma
from numpy import array, ptp, where, logical_and, power, sin, pi
from numpy.random import uniform, normal, choice
from time import time
from copy import deepcopy
from config import Config
from model.scheduler.fitness import Fitness
from utils.schedule_util import matrix_to_schedule
from sys import exit
import matplotlib.pyplot as plt
import math


class Root:
    """
        Base class for all models (single or multiple objective)
    """
    ID_POS = 0
    ID_FIT = 1
    ID_LOCAL_POS = 2       # Personal best location
    ID_LOCAL_FIT = 3       # Personal best fitness

    EPSILON = 10E-10

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True):
        self.problem = problem
        self.pop_size = pop_size
        self.epoch = epoch
        self.func_eval = func_eval
        self.lb = lb
        self.ub = ub
        self.verbose = verbose
        self.Fit = Fitness(problem)
        self.n_objs = 1

    def create_solution(self):
        while True:
            matrix = uniform(self.lb, self.ub, self.problem["shape"])
            schedule = matrix_to_schedule(self.problem, matrix.astype(int))
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                break
        return [matrix, fitness, matrix, fitness]        # [solution, fit]

    def early_stopping(self, array, patience=5):
        if patience <= len(array) - 1:
            value = array[len(array) - patience]
            arr = array[len(array) - patience + 1:]
            check = 0
            for val in arr:
                if val < value:
                    check += 1
            if check != 0:
                return False
            return True
        raise ValueError

    def get_index_roulette_wheel_selection(self, list_fitness: list):
        """ It can handle negative also. Make sure your list fitness is 1D-numpy array"""
        list_fitness = array(list_fitness)
        scaled_fitness = (list_fitness - min(list_fitness)) / ptp(list_fitness)
        minimized_fitness = 1.0 - scaled_fitness
        total_sum = sum(minimized_fitness)
        r = uniform(low=0, high=total_sum)
        for idx, f in enumerate(minimized_fitness):
            r = r + f
            if r > total_sum:
                return idx

    def get_indexes_k_tournament_selection(self, list_fitness: list, k=10):
        list_fitness = array(list_fitness)
        idx_list = choice(range(0, len(list_fitness)), k, replace=False)
        idx_list = [[idx, list_fitness[idx]] for idx in idx_list]
        idx_list = sorted(idx_list, key=lambda item: item[1])       # Sort by fitness
        return idx_list[0][0], idx_list[1][0]                       # Select two best fitness

    def amend_position_random(self, position=None):
        return where(logical_and(self.lb <= position, position <= self.ub), position, uniform(self.lb, self.ub))

    def get_current_worst(self, pop=None):
        if isinstance(pop, dict):
            pop_temp = deepcopy(pop.values())
        elif isinstance(pop, list):
            pop_temp = deepcopy(pop)
        else:
            exit()
        if Config.METRICS in Config.METRICS_MAX:
            current_worst = min(pop_temp, key=lambda x: x[self.ID_FIT])
        else:
            current_worst = max(pop_temp, key=lambda x: x[self.ID_FIT])
        return deepcopy(current_worst)

    def get_current_best(self, pop=None):
        if isinstance(pop, dict):
            pop_temp = deepcopy(pop.values())
        elif isinstance(pop, list):
            pop_temp = deepcopy(pop)
        else:
            exit()
        if Config.METRICS in Config.METRICS_MAX:
            current_best = max(pop_temp, key=lambda x: x[self.ID_FIT])
        else:
            current_best = min(pop_temp, key=lambda x: x[self.ID_FIT])
        return deepcopy(current_best)

    def update_old_solution(self, old_solution, new_solution):
        if Config.METRICS in Config.METRICS_MAX:
            if new_solution[self.ID_FIT] > old_solution[self.ID_FIT]:
                return new_solution
        else:
            if new_solution[self.ID_FIT] < old_solution[self.ID_FIT]:
                return new_solution
        return old_solution

    def update_old_population(self, pop_old:list, pop_new:list):
        for i in range(0, self.pop_size):
            if Config.METRICS in Config.METRICS_MAX:
                if pop_new[i][self.ID_FIT] > pop_old[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
            else:
                if pop_new[i][self.ID_FIT] < pop_old[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
        return pop_old

    def adding_element_to_dict(self, obj: dict, key: list, value: list):
        for idx, k in enumerate(key):
            obj[k].append(value[idx])
        return obj

    def check_objective(self):
        if Config.METRICS == "pareto" and self.__class__.__name__ not in Config.MULTI_OBJECTIVE_SUPPORTERS:
            print(f'Method: {self.__class__.__name__} doesn"t support pareto-front fitness function type')
            exit()
        if self.verbose:
            print(f'Start training by: {self.__class__.__name__} algorithm, fitness type: {Config.METRICS}')

    def check_log(self):
        logline = f'with time-bound: {Config.TIME_BOUND_VALUE_PER_TASK * self.problem["n_tasks"]} seconds.' if Config.TIME_BOUND_KEY else "without time-bound."
        if Config.MODE == "epoch":
            logline = f"Training by: epoch (mode) with: {self.epoch} epochs, {logline}"
        elif Config.MODE == "fe":
            logline = f'Training by: function evalution (mode) with: {self.func_eval} FE, {logline}'
        if self.verbose:
            print(logline)

    def check_break_loop(self, mode_value, current_best, g_best, time_end, time_bound_start):
        break_loop = False
        if self.verbose:
            print(f'{Config.MODE.upper()}: {mode_value}, Current best fit: {current_best[self.ID_FIT]:.4f}, '
                  f'Global best fit {g_best[self.ID_FIT]:.4f}, Time: {time_end:.2f} seconds')
        if Config.TIME_BOUND_KEY:
            if time() - time_bound_start >= Config.TIME_BOUND_VALUE_PER_TASK * self.problem["n_tasks"]:
                print('====== Over time for training {} ======'.format(self.problem["n_tasks"]))
                break_loop = True
        return break_loop

    def check_break_loop_multi(self, mode_value, front0, pop, time_end, time_bound_start):
        break_loop = False
        if self.verbose:
            # obj = [zeros(len(pop)) for i in range(self.n_objs)]
            # for idx, item in enumerate(pop.values()):
            #     for i in range(self.n_objs):
            #         obj[i][idx] = float(item[self.ID_FIT][i])
            # visualize_2D(obj[:2])
            print(f'{Config.MODE.upper()}: {mode_value}, Front size: {len(front0[0])}, including {list(pop.values())[front0[0][0]][self.ID_FIT]}, '
                  f'time: {time_end:.2f} seconds')
        if Config.TIME_BOUND_KEY:
            if time() - time_bound_start >= Config.TIME_BOUND_VALUE_PER_TASK * self.problem["n_tasks"]:
                print('====== Over time for training ======')
                break_loop = True
        return break_loop

    def get_g_best(self, pop):
        if Config.METRICS in Config.METRICS_MAX:
            g_best = max(pop, key=lambda x: x[self.ID_FIT])
        else:
            g_best = min(pop, key=lambda x: x[self.ID_FIT])
        return g_best

    def update_g_best_get_current_best(self, pop, g_best):
        if Config.METRICS in Config.METRICS_MAX:
            current_best = max(pop, key=lambda x: x[self.ID_FIT])
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
        else:
            current_best = min(pop, key=lambda x: x[self.ID_FIT])
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
        return g_best, current_best

    def get_step_levy_flight(self, beta=1.0, step=0.001):
        """
        Parameters
        ----------
        epoch (int): current iteration
        position : 1-D numpy array
        g_best_position : 1-D numpy array
        step (float, optional): 0.001
        case (int, optional): 0, 1, 2

        """
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = power(gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = normal(0, sigma_muy ** 2)
        v = normal(0, sigma_v ** 2)
        s = muy / power(abs(v), 1 / beta)

        return step * s

        # levy = uniform(self.lb, self.ub) * step * s * (position - g_best_position)
        #
        # if case == 0:
        #     return levy
        # elif case == 1:
        #     return position + 1.0 / sqrt(epoch + 1) * sign(random() - 0.5) * levy
        # elif case == 2:
        #     return position + normal(0, 1, len(self.lb)) * levy
        # elif case == 3:
        #     return position + 0.01 * levy


    def step_decay(self, epoch, init_er):
       init_explore_rate = init_er
       drop = (1 - 1 / (math.e + 3)) 
       epochs_drop = math.floor(math.sqrt(self.epoch))
       explore_rate = init_explore_rate * math.pow(drop, math.floor((1 + epoch)/epochs_drop))
       return max(explore_rate, 0.02)
   
    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        pass

    def train(self):
        time_total = time()
        time_bound_start = time()
        self.check_objective()
        self.check_log()

        ## Initialization
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_g_best(pop)
        g_best_list = [g_best[self.ID_FIT]]
        current_best_list = []

        if Config.MODE == 'epoch':
            for epoch in range(self.epoch):
                # print(epoch)
                time_epoch_start = time()
                pop = self.evolve(pop, None, epoch, g_best)
                g_best, current_best = self.update_g_best_get_current_best(pop, g_best)
                g_best_list.append(g_best[self.ID_FIT])
                current_best_list.append(current_best[self.ID_FIT])
                print("EPOCH:", epoch, " / ", round(current_best[self.ID_FIT], 3))
                # plt.plot(current_best_list)
                # ft = [pop[i][self.ID_FIT] for i in range(self.pop_size)]
                # plt.plot(ft, 'o')
                # plt.show()
                time_epoch_end = time() - time_epoch_start
                break_loop = self.check_break_loop(epoch+1, current_best, g_best, time_epoch_end, time_bound_start)
                if break_loop:                
                    print("LAST EPOCH:", epoch, " / ", round(current_best[self.ID_FIT], 3))
                    break
            time_total = time() - time_total
            return g_best[0], g_best[1], array(g_best_list), time_total
        elif Config.MODE == "fe":
            fe_counter = 0
            time_fe_start = time()
            while fe_counter < self.func_eval:
                pop, counter = self.evolve(pop, Config.MODE, None, g_best)
                g_best, current_best = self.update_g_best_get_current_best(pop, g_best)
                g_best_list.append(g_best[self.ID_FIT])
                current_best_list.append(current_best[self.ID_FIT])
                print("EPOCH:", epoch, " / ", round(current_best[self.ID_FIT], 3))
                fe_counter += counter
                time_fe_end = time() - time_fe_start
                break_loop = self.check_break_loop(fe_counter, current_best, g_best, time_fe_end, time_bound_start)
                if break_loop:
                    break
            time_total = time() - time_total
            return g_best[0], g_best[1], array(g_best_list), time_total


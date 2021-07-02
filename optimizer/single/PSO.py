#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:32, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from config import Config
from optimizer.root import Root
from numpy.random import uniform, rand
from numpy import array, zeros, mean, stack, ones
from numpy import min as np_min
from numpy import max as np_max
from copy import deepcopy
from time import time
from utils.schedule_util import matrix_to_schedule


class BasePSO(Root):
    ID_POS = 0              # Current position
    ID_FIT = 1              # Current fitness
    ID_VEL = 2              # Current velocity
    ID_LOCAL_POS = 3        # Personal best location
    ID_LOCAL_FIT = 4        # Personal best fitness

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"w_min": 0.4, "w_max": 0.9, "c_local": 1.2, "c_global": 2}
        self.w_min = paras["w_min"]
        self.w_max = paras["w_max"]
        self.c_local = paras["c_local"]
        self.c_global = paras["c_global"]

    def create_solution(self):
        while True:
            pos = uniform(self.lb, self.ub, self.problem["shape"])
            schedule = matrix_to_schedule(self.problem, pos.astype(int))
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                vel = uniform(self.lb, self.ub, self.problem["shape"])
                break
        return [pos, fitness, vel, pos, fitness]
        # [solution, fit, velocity, local_solution, local_fitness]

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # Update weight after each move count  (weight down)
        w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min

        for i in range(self.pop_size):
            while True:
                v_new = w * pop[i][self.ID_VEL] + self.c_local * uniform() * (pop[i][self.ID_LOCAL_POS] - pop[i][self.ID_POS]) + \
                        self.c_global * uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                x_new = pop[i][self.ID_POS] + v_new  # Xi(new) = Xi(old) + Vi(neID_POSw) * deltaT (deltaT = 1)
                # v_new = self.amend_position_random(v_new)
                x_new = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, x_new.astype(int))
                if schedule.is_valid():
                    fit_new = self.Fit.fitness(schedule)
                    pop[i][self.ID_POS] = x_new
                    pop[i][self.ID_FIT] = fit_new
                    pop[i][self.ID_VEL] = v_new
                    # Update current position, current velocity and compare with past position, past fitness (local best)
                    if Config.METRICS in Config.METRICS_MAX:
                        if fit_new > pop[i][self.ID_LOCAL_FIT]:
                            pop[i][self.ID_LOCAL_POS] = x_new
                            pop[i][self.ID_LOCAL_FIT] = fit_new
                    else:
                        if fit_new < pop[i][self.ID_LOCAL_FIT]:
                            pop[i][self.ID_LOCAL_POS] = x_new
                            pop[i][self.ID_LOCAL_FIT] = fit_new
                    break

        if fe_mode is None:
            return pop
        else:
            counter = self.pop_size  # pop_new + pop_mutation operations
            return pop, counter


class CPSO(Root):
    """
            CPSO: Chaos Particle Swarm Optimization
        Paper: Improved particle swarm optimization combined with chaos
    """

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"w_min": 0.4, "w_max": 0.9, "c_local": 0.2, "c_global": 1.2}
        self.w_min = paras["w_min"]
        self.w_max = paras["w_max"]
        self.c_local = paras["c_local"]
        self.c_global = paras["c_global"]
        self.lb = ones(self.problem["shape"]) * lb
        self.ub = ones(self.problem["shape"]) * ub

    def __get_weights__(self, fit, fit_avg, fit_min):
        if fit <= fit_avg:
            return self.w_min + (self.w_max - self.w_min) * (fit - fit_min) / (fit_avg - fit_min)
        else:
            return self.w_max

    def self_evolve(self, pop=None, fe_mode=None, g_best=None, v_list=None, pop_local=None, n_locals=10):
        r = rand()
        list_fits = [item[self.ID_FIT] for item in pop]
        fit_avg = mean(list_fits)
        fit_min = np_min(list_fits)

        for i in range(self.pop_size):
            w = self.__get_weights__(pop[i][self.ID_FIT], fit_avg, fit_min)
            while True:
                v_new = w * v_list[i] + self.c_local * rand() * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS]) + \
                        self.c_global * rand() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                x_new = pop[i][self.ID_POS] + v_new
                x_new = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, x_new.astype(int))
                if schedule.is_valid():
                    fit_new = self.Fit.fitness(schedule)
                    pop[i] = [x_new, fit_new]
                    # Update current position, current velocity and compare with past position, past fitness (local best)
                    if Config.METRICS in Config.METRICS_MAX:
                        if fit_new > pop_local[i][self.ID_FIT]:
                            pop_local[i] = [x_new, fit_new]
                    else:
                        if fit_new < pop_local[i][self.ID_FIT]:
                            pop_local[i] = [x_new, fit_new]
                    break
        _, g_best = self.update_g_best_get_current_best(pop, g_best)        # g_best, current_best

        ## Implement chaostic local search for the best solution
        cx_best_0 = (g_best[self.ID_POS] - self.lb) / (self.ub - self.lb)  # Eq. 7
        cx_best_1 = 4 * cx_best_0 * (1 - cx_best_0)  # Eq. 6
        x_best = self.lb + cx_best_1 * (self.ub - self.lb)  # Eq. 8
        schedule = matrix_to_schedule(self.problem, x_best.astype(int))
        if schedule.is_valid():
            fit_new = self.Fit.fitness(schedule)
            if Config.METRICS in Config.METRICS_MAX:
                if fit_new > g_best[self.ID_FIT]:
                    g_best = [x_best, fit_new]
            else:
                if fit_new < g_best[self.ID_FIT]:
                    g_best = [x_best, fit_new]

        # bound_min = stack([self.lb, g_best[self.ID_POS] - r * (self.ub - self.lb)])
        # self.lb = np_max(bound_min, axis=0)
        # bound_max = stack([self.ub, g_best[self.ID_POS] + r * (self.ub - self.lb)])
        # self.ub = np_min(bound_max, axis=0)

        pop_new_child = [self.create_solution() for _ in range(self.pop_size - n_locals)]
        pop_new = sorted(pop, key=lambda item: item[self.ID_FIT])
        pop = pop_new[:n_locals] + pop_new_child

        if fe_mode is None:
            return pop
        else:
            counter = self.pop_size  # pop_new + pop_mutation operations
            return pop, counter

    def train(self):
        time_total = time()
        time_bound_start = time()
        self.check_objective()
        self.check_log()

        ## Initialization
        N_CLS = int(self.pop_size / 5)  # Number of chaotic local searches

        pop = [self.create_solution() for _ in range(self.pop_size)]
        v_max = 0.5 * (self.ub - self.lb)
        v_min = zeros(self.problem["shape"])
        v_list = [uniform(v_min, v_max, self.problem["shape"]) for _ in range(self.pop_size)]
        pop_local = deepcopy(pop)
        g_best = self.get_g_best(pop)
        g_best_list = [g_best[self.ID_FIT]]

        if Config.MODE == 'epoch':
            for epoch in range(self.epoch):
                time_epoch_start = time()
                pop = self.self_evolve(pop, None, g_best, v_list, pop_local, N_CLS)
                g_best, current_best = self.update_g_best_get_current_best(pop, g_best)
                g_best_list.append(g_best[self.ID_FIT])
                time_epoch_end = time() - time_epoch_start
                break_loop = self.check_break_loop(epoch + 1, current_best, g_best, time_epoch_end, time_bound_start)
                if break_loop:
                    break
            time_total = time() - time_total
            return g_best[0], g_best[1], array(g_best_list), time_total
        elif Config.MODE == "fe":
            fe_counter = 0
            time_fe_start = time()
            while fe_counter < self.func_eval:
                pop, counter = self.self_evolve(pop, Config.MODE, g_best, v_list, pop_local, N_CLS)
                g_best, current_best = self.update_g_best_get_current_best(pop, g_best)
                g_best_list.append(g_best[self.ID_FIT])
                fe_counter += counter
                time_fe_end = time() - time_fe_start
                break_loop = self.check_break_loop(fe_counter, current_best, g_best, time_fe_end, time_bound_start)
                if break_loop:
                    break
            time_total = time() - time_total
            return g_best[0], g_best[1], array(g_best_list), time_total

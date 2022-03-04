#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 12:50, 17/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from copy import deepcopy
from optimizer.root import Root
from numpy.random import uniform, normal, random
from numpy import ceil, sqrt, array, mean, cos, std, zeros,  subtract, sum
import numpy as np
from utils.schedule_util import matrix_to_schedule
from random import choice, sample, randint
import math
from bisect import bisect_left


class VNS(Root):
    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        self.n_sqrt = int(ceil(sqrt(self.pop_size)))
        self.n_half = int(pop_size/2)
        self.n_sqrt = int(ceil(sqrt(self.pop_size)))
        
   
    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # epoch: current chance, self.epoch: number of chances
        wf = 0.5   # weight factor
        pop = sorted(pop, key=lambda x: x[self.ID_FIT])
        new_pop = deepcopy(pop)
        wf = self.step_decay(epoch, 0.5)
        K = self.n_sqrt
        for i in range(1, self.pop_size):
            best_fit = np.inf
            for _ in range(K):
                temp = pop[i][self.ID_POS] + normal(0, 0.05, self.problem["shape"])
                child = self.amend_position_random(temp)
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    if best_fit > fit:
                        best_fit = fit
                        best = temp
                        
                new_pop[i][self.ID_POS] = best
                
        for i in range(self.pop_size):
            if new_pop[i][self.ID_FIT] < pop[i][self.ID_FIT]:
                pop[i] = new_pop[i]
                
        current_best = self.get_current_best(pop)
        # Decomposition
        ## Eq. 10, 11, 12, 9
        for i in range(0, self.pop_size):
            while True:
                r3 = uniform()
                d = normal(0, 0.5)
                e = r3 * randint(1, 3) - 1
                h = 2 * r3 - 1
                child = current_best[self.ID_POS] + d * (e * current_best[self.ID_POS] - h * pop[i][self.ID_POS])

                for j in range(len(current_best[self.ID_POS])):
                        current_best[self.ID_POS][j] += self.get_step_levy_flight()

                child = self.amend_position_random(child)
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    if fit < new_pop[i][self.ID_FIT]:
                        new_pop[i][self.ID_POS] = child
                        new_pop[i][self.ID_FIT] = fit
                        if new_pop[i][self.ID_FIT] < pop[i][self.ID_FIT] or random() < 0.3:
                            pop[i] = new_pop[i]
                    break 
        # pop = new_pop
        if fe_mode is None:
            return pop
        else:
            counter = 2 * self.pop_size  # pop_new + pop_mutation operations
            return pop, counter
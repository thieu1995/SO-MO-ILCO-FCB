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
from numpy import ceil, sqrt, array, mean, cos, std
from utils.schedule_util import matrix_to_schedule
from random import choice, sample, randint
import math


class BaseLCO(Root):
    """
    The original version of: Life Choice-based Optimization (LCO)
        (A novel life choice-based optimizer)
    Link:
        DOI: https://doi.org/10.1007/s00500-019-04443-z
    """
    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"r1": 2.35}
        self.r1 = paras["r1"]
        self.n_sqrt = int(ceil(sqrt(self.pop_size)))

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # epoch: current chance, self.epoch: number of chances
        for i in range(0, self.pop_size):
            while True:
                rand_number = random()

                if rand_number > 0.875:  # Update using Eq. 1, update from n best position
                    temp = array([random() * pop[j][self.ID_POS] for j in range(0, self.n_sqrt)])
                    temp = mean(temp, axis=0)
                elif rand_number < 0.7:  # Update using Eq. 2-6
                    f1 = 1 - epoch / self.epoch
                    f2 = 1 - f1
                    if i == 0:
                        pop[i] = deepcopy(g_best)
                        continue
                    else:
                        best_diff = f1 * self.r1 * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                        better_diff = f2 * self.r1 * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS])
                    temp = pop[i][self.ID_POS] + random() * better_diff + random() * best_diff
                else:
                    temp = self.ub - (pop[i][self.ID_POS] - self.lb) * random()

                child = self.amend_position_random(temp)
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    break
            pop[i] = [child, fit]

        if fe_mode is None:
            return pop
        else:
            counter = self.pop_size  # pop_new + pop_mutation operations
            return pop, counter


class I_LCO(Root):
    """
    My version of: Improved Life Choice-based Optimization (ILCO)
    Link:
        DOI: https://doi.org/10.1007/s00500-019-04443-z
    """

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        self.n_sqrt = int(ceil(sqrt(self.pop_size)))
        self.n_half = int(pop_size/2)
        if paras is None:
            paras = {"r1": 2.35}
        self.r1 = paras["r1"]
        self.n_sqrt = int(ceil(sqrt(self.pop_size)))
        
   
    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # epoch: current chance, self.epoch: number of chances
        # wf = 0.5 + 0.5 * (epoch / self.epoch) ** 2   # weight factor
        a = self.step_decay(epoch)
        pop = sorted(pop, key=lambda x: x[self.ID_FIT], reverse=True)
        # print(pop[-1][self.ID_POS])
        # print([x[self.ID_FIT] for x in pop])
        for i in range(0, self.pop_size - 1):
            while True:
                rand_number = random()
                if rand_number > 0.9:  # Update using Eq. 1, update from n best position
                    rd_index = sample([i + (self.pop_size - self.n_sqrt) for i in range (self.n_sqrt)], 3)
                    temp = array([pop[j][self.ID_POS] for j in rd_index])
                    coeff = random() * 0.5 * a
                    temp = coeff * mean(temp, axis=0) + (1 - coeff) * pop[i][self.ID_POS]
                elif rand_number < 0.7:  # Update using Eq. 2-6
                    if random() < 0.3:
                        better_diff = a * (pop[randint(i, self.pop_size - 1)][self.ID_POS] - pop[i][self.ID_POS]) 
                        best_diff = (1 - a) * (pop[i][self.ID_LOCAL_POS] - pop[i][self.ID_POS])
                        temp = pop[i][self.ID_POS] + (better_diff + best_diff)
                    else:
                        temp = pop[i][self.ID_POS] + a * \
                            (g_best[self.ID_POS] - pop[i][self.ID_POS] + normal(0, 0.5, self.problem["shape"]))

                else:  # Exploration, update group 3
                    pos_new = self.ub - (pop[i][self.ID_POS] - self.lb) * random()
                    if random() < 0.5:
                        pos_new = self.lb + self.ub - g_best[self.ID_POS] + random() * (g_best[self.ID_POS] - pos_new)
                    temp = pos_new
                child = self.amend_position_random(temp)
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    break
            if fit < pop[i][self.ID_LOCAL_FIT]:
                pop[i][self.ID_POS] = child
                pop[i][self.ID_FIT] = fit
                pop[i][self.ID_LOCAL_POS] = child
                pop[i][self.ID_LOCAL_FIT] = fit
            else:
                pop[i][self.ID_POS] = child
                pop[i][self.ID_FIT] = fit
                
            
        if fe_mode is None:
            return pop
        else:
            counter = self.pop_size  # pop_new + pop_mutation operations
            return pop, counter

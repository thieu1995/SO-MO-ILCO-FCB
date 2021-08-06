#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:32, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from config import Config
from optimizer.root_MOPSORI import Root3
import random
from utils.schedule_util import matrix_to_schedule
from numpy.random import uniform, randint, normal, random
from numpy import ceil, sqrt, array, mean, cos
from random import choice, sample, randint
from uuid import uuid4
import numpy as np
from copy import deepcopy


class MO_IBLA(Root3):
    
    """
    My version of: Improved Life Choice-based Optimization (ILCO)
    Link:
        DOI: https://doi.org/10.1007/s00500-019-04443-z
    """

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"r1": 2.35}
        self.r1 = paras["r1"]
        self.n_sqrt = int(ceil(sqrt(self.pop_size)))
        self.n_half = int(pop_size/2)

    def get_gbest_mean(self, g_best):
        temp = array([item[self.ID_POS] for item in g_best])
        return mean(temp, axis=0)

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # epoch: current chance, self.epoch: number of chances
        wf = 0.5   # weight factor
        new_pop = deepcopy(pop)
        wf = self.step_decay2(epoch, 0.5)
        K = self.n_sqrt
        n_l = self.n_sqrt
        key_list = list(pop.keys())
        order = {}
        fronts, rank = self.fast_non_dominated_sort(pop)
        for i in range(self.pop_size):
            order[key_list[i]] = rank[i]
        # print(key_list)
        key_list = sorted(key_list, key = lambda x: order[x])
        # print(key_list)
        master = sum(array([pop[key_list[j]][self.ID_POS] for j in range(K)]))
        for i in range(1, self.pop_size):
            idx = key_list[i]
            while True:
                if i < K:
                    temp = (master - pop[idx][self.ID_POS]) / (K - 1) - pop[idx][self.ID_POS]
                    new_pop[idx][self.ID_POS] = pop[idx][self.ID_POS] + (temp + normal(0, 0.1, self.problem["shape"]))
    
                    best = new_pop[idx][self.ID_POS]
                    best_fit = 1e9
                    child = self.amend_position_random(new_pop[idx][self.ID_POS])
                    schedule = matrix_to_schedule(self.problem, child.astype(int))
                    if schedule.is_valid():
                        best_fit = self.Fit.fitness(schedule)
                        
                    for j in range(n_l):
                        temp = new_pop[idx][self.ID_POS] + normal(0, 0.1, self.problem["shape"])
                        child = self.amend_position_random(temp)
                        schedule = matrix_to_schedule(self.problem, child.astype(int))
                        if schedule.is_valid():
                            fit = self.Fit.fitness(schedule)
                            if self.is_better(best_fit, fit):
                                best_fit = fit
                                best = temp
                                
                    new_pop[idx][self.ID_POS] = best
                else:
                    rand_number = random()
                    if rand_number < 0.9:
                        d = normal(0, 0.1)
                        teacher = np.argmin([sum(np.absolute(np.subtract(pop[key_list[i]][self.ID_POS], pop[key_list[j]][self.ID_POS]))) for j in range (K) if j != i]) 
                        new_pop[key_list[i]][self.ID_POS] = pop[idx][self.ID_POS] +\
                            d * (pop[key_list[teacher]][self.ID_POS] - pop[idx][self.ID_POS])  \
                                + normal(0, 0.01, self.problem["shape"])
                    else:
                        # pr = random()
                        # _pos = bisect_left(prob, pr)
                        _pos  = randint(0, self.pop_size - 1)
                        friend = new_pop[key_list[_pos]][self.ID_POS]
                        new_pop[idx][self.ID_POS] = wf * friend + (1 - wf) * new_pop[idx][self.ID_POS] \
                            + normal(0, 0.01, self.problem["shape"])
                        
                child = self.amend_position_random(new_pop[idx][self.ID_POS])
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    new_pop[idx][self.ID_POS] = child
                    new_pop[idx][self.ID_FIT] = fit
                    break
                
            
        ## find current best used in decomposition
        current_best = pop[key_list[0]]
        for i in range(self.pop_size):
            idx = key_list[i]
            if self.is_better(new_pop[idx][self.ID_FIT],pop[idx][self.ID_FIT]) or random() < 0.1:
                pop[idx] = new_pop[idx]
            if self.is_better(pop[idx][self.ID_FIT], current_best[self.ID_FIT]):
                current_best[self.ID_POS] = deepcopy(pop[idx][self.ID_POS])
                
        # Decomposition
        ## Eq. 10, 11, 12, 9
        for i in range(0, self.pop_size):
            idx = key_list[i]
            while True:
                r3 = uniform()
                d = normal(0, 0.5)
                e = r3 * randint(1, 3) - 1
                h = 2 * r3 - 1
                child = current_best[self.ID_POS] + d * (e * current_best[self.ID_POS] - h * pop[idx][self.ID_POS])

                current_best[self.ID_POS] = current_best[self.ID_POS] + normal(0, 0.03, self.problem["shape"])

                child = self.amend_position_random(child)
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    if self.is_better(fit, new_pop[idx][self.ID_FIT]):
                        new_pop[idx][self.ID_POS] = child
                        new_pop[idx][self.ID_FIT] = fit
                    break
        ## Update old population
        # pop = self.update_old_population(pop, new_pop)
        
        for i in range(self.pop_size):
            idx = key_list[i]
            if self.is_better(new_pop[idx][self.ID_FIT], pop[idx][self.ID_FIT]) or random() < 0.3:
                pop[idx] = new_pop[idx]
                
        # pop = new_pop
        if fe_mode is None:
            return pop
        else:
            counter = 2 * self.pop_size  # pop_new + pop_mutation operations
            return pop, counter
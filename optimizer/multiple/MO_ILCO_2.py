#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:32, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from config import Config
from copy import deepcopy
import numpy as np
from optimizer.root_MOPSORI import Root3
from bisect import bisect_left
from numpy import ceil, sqrt, array, mean, cos, std, zeros,  subtract, sum
import random
from utils.schedule_util import matrix_to_schedule
from numpy.random import uniform, randint, normal, random
from numpy import ceil, sqrt, array, mean, cos
from random import choice, sample, randint


class MO_ILCO_2(Root3):
    
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
        self.prob = [np.exp((self.pop_size - i + self.n_sqrt) / self.n_sqrt) for i in range(self.pop_size)]
        self.prob = array(self.prob / sum(self.prob))
        for i in range(1, self.pop_size):
            self.prob[i] += self.prob[i - 1]

    def get_gbest_mean(self, g_best):
        temp = array([item[self.ID_POS] for item in g_best])
        return mean(temp, axis=0) 
    
    def get_current_best(self, pop):
        key_list = list(pop.keys())
        idx = key_list[0]
        for i in range(1, self.pop_size):
            if self.is_dominate(pop[key_list[i]][self.ID_FIT], pop[idx][self.ID_FIT]):
                idx = key_list[i]
        return pop[idx]
   
    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # epoch: current chance, self.epoch: number of chances
        key_list = list(pop.keys())
        id_list = {}
        for i in range(0, self.pop_size):
            id_list[key_list[i]] = i
        fronts, rank = self.fast_non_dominated_sort(pop)
        n_best = len(fronts)
            
        key_list = sorted(key_list, key=lambda x: id_list[x])
        new_pop = deepcopy(pop)
        wf = self.step_decay(epoch, 0.5)
        
        for _i in range(0, self.pop_size):
            i = key_list[_i]
            while True:
                K = min(n_best, bisect_left(self.prob, uniform()) + 1)
                # K = randint(1, self.n_sqrt)
                master = sum(array([pop[key_list[j]][self.ID_POS] for j in sample([_ for _ in range(K)], min(K, 3))]), axis=0) / min(K, 3)
                if _i < K:
                    temp = pop[i][self.ID_POS] + normal(0, 0.3) * (master - pop[i][self.ID_POS])
                    for j in range(len(pop[i][self.ID_POS])):
                        temp[j] += self.get_step_levy_flight()
                    best = new_pop[i][self.ID_POS]
                    best_fit = 1e9
                    child = self.amend_position_random(new_pop[i][self.ID_POS])
                    schedule = matrix_to_schedule(self.problem, child.astype(int))
                    if schedule.is_valid():
                        best_fit = self.Fit.fitness(schedule)
                        
                    for j in range(self.n_sqrt):
                        for k in range(self.problem["shape"]):
                            temp[k] += wf * normal(0, 0.1)
                        child = self.amend_position_random(temp)
                        schedule = matrix_to_schedule(self.problem, child.astype(int))
                        if schedule.is_valid():
                            fit = self.Fit.fitness(schedule)
                            if self.is_dominate(fit, best_fit):
                                best_fit = fit
                                best = temp
                                
                    new_pop[i][self.ID_POS] = best
                else:
                    rand_number = random()
                    if rand_number > 0.5:
                        d = normal(0, 0.3)
                        teacher = np.argmin([sum(np.absolute(subtract(pop[i][self.ID_POS], pop[key_list[j]][self.ID_POS]))) for j in range (K) if j != i]) 
                        new_pop[i][self.ID_POS] = pop[i][self.ID_POS] +\
                            d * (pop[key_list[teacher]][self.ID_POS] - pop[i][self.ID_POS])
                        for j in range(len(new_pop[i][self.ID_POS])):
                            new_pop[i][self.ID_POS][j] += self.get_step_levy_flight()
                    elif rand_number < 0.4:
                        _pos  = randint(0, _i - 1)
                        friend = new_pop[key_list[_pos]][self.ID_POS]
                        new_pop[i][self.ID_POS] = wf * friend + (1 - wf) * new_pop[i][self.ID_POS] \
                             + self.get_step_levy_flight()
                    else:
                        pos_new = pop[i][self.ID_POS] + (self.ub - (pop[i][self.ID_POS] - self.lb) - pop[i][self.ID_POS]) * random()
                        new_pop[i][self.ID_POS] = pos_new
                        
                child = self.amend_position_random(new_pop[i][self.ID_POS])
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    new_pop[i][self.ID_POS] = child
                    new_pop[i][self.ID_FIT] = fit
                    if self.is_dominate(new_pop[i][self.ID_FIT], pop[i][self.ID_FIT])\
                        or (self.is_non_dominated(pop[i][self.ID_FIT], new_pop[i][self.ID_FIT])
                            and random() < 0.1):
                        pop[i] = new_pop[i]
                    break
                 
        current_best = self.get_current_best(pop)
        # Decomposition
        ## Eq. 10, 11, 12, 9
        for _i in range(0, self.pop_size):
            i = key_list[_i]
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
                    if self.is_dominate(fit, new_pop[i][self.ID_FIT]):
                        new_pop[i][self.ID_POS] = child
                        new_pop[i][self.ID_FIT] = fit
                        if self.is_dominate(new_pop[i][self.ID_FIT], pop[i][self.ID_FIT])\
                            or (self.is_non_dominated(pop[i][self.ID_FIT], new_pop[i][self.ID_FIT])
                                and random() < 0.1):
                            pop[i] = new_pop[i]
                    break 
                
        # pop = new_pop
        if fe_mode is None:
            return pop
        else:
            counter = 2 * self.pop_size  # pop_new + pop_mutation operations
            return pop, counter

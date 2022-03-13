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
from utils.schedule_util import matrix_to_schedule
from random import choice, sample, randint
from bisect import bisect_left
import numpy as np
import math

class ILCO_a1(Root):
    """
    My version of: Improved Life Choice-based Optimization (ILCO)
    Link:
        DOI: https://doi.org/10.1007/s00500-019-04443-z
    """

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        self.n_sqrt = int(ceil(sqrt(self.pop_size)))
        self.n_half = int(pop_size/2)
        self.n_sqrt = int(ceil(sqrt(self.pop_size)))
        if paras is None:
            paras = {"r1": 2.35}
        self.r1 = paras["r1"]
        
        self.prob = [np.exp((self.pop_size - i + self.n_sqrt) / self.n_sqrt) for i in range(self.pop_size)]
        self.prob = array(self.prob / sum(self.prob))
        for i in range(1, self.pop_size):
            self.prob[i] += self.prob[i - 1]
   
    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # epoch: current chance, self.epoch: number of chances
        pop = sorted(pop, key=lambda x: x[self.ID_FIT])
        new_pop = deepcopy(pop)
        wf = self.step_decay(epoch, 0.5)
        
        for i in range(1, self.pop_size):
            while True:
                K = min(self.n_sqrt, bisect_left(self.prob, uniform()) + 1)
                # K = randint(1, self.n_sqrt)
                master = sum(array([pop[j][self.ID_POS] for j in sample([_ for _ in range(K)], min(K, 3))]), axis=0) / min(K, 3)
                if i < K:
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
                            if best_fit > fit:
                                best_fit = fit
                                best = temp
                                
                    new_pop[i][self.ID_POS] = best
                else:
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
                child = self.amend_position_random(new_pop[i][self.ID_POS])
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    new_pop[i][self.ID_POS] = child
                    new_pop[i][self.ID_FIT] = fit
                    if new_pop[i][self.ID_FIT] < pop[i][self.ID_FIT] or random() < 0.1:
                        pop[i] = new_pop[i]
                    break
                 
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
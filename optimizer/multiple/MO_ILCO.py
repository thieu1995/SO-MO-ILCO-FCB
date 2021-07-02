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


class MO_ILCO(Root3):
    
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
        wf = 0.5 + 0.5 * (epoch / self.epoch) ** 2   # weight factor
        a = 1.0 - cos(0) * (1.0 / cos(1 - (epoch + 1) / self.epoch))
        coef = 2 - 2 * epoch / (self.epoch - 1)  # linearly decreased from 2 to 0

        for i in range(0, self.pop_size):
            idx = list(pop.keys())[i]
            
            while True:
                rand_number = random()

                if rand_number < 0.1:  # Update using Eq. 1, update from n best position
                    temp = array([random() * pop[list(pop.keys())[randint(0, self.pop_size - 1)]][self.ID_POS] \
                                  for j in range(0, self.n_sqrt)], dtype = object)
                    temp = mean(temp, axis=0)
                elif 0.1 <= rand_number <= 0.75:  # Exploitation, update group 2
                    g_best_mean = self.get_gbest_mean(g_best)
                    if random() < 0.5:
                        if i != 0:
                            better_diff = a * (pop[list(pop.keys())[i - 1]][self.ID_POS] - pop[idx][self.ID_POS])
                        else:
                            better_diff = a * (g_best_mean - pop[idx][self.ID_POS])
                        best_diff = (1 - a) * (g_best_mean - pop[idx][self.ID_POS])
                        temp = wf * pop[idx][self.ID_POS] + uniform() * better_diff + uniform() * best_diff
                    else:
                        # temp = wf * self.get_gbest_mean(g_best) + \
                        #        normal(0, 1, self.problem["shape"]) * a * (self.get_gbest_mean(g_best) - pop[idx][self.ID_POS])
                        temp = a * g_best_mean + coef * self.get_step_levy_flight(beta=0.5, step=0.01) * (g_best_mean - uniform() * pop[i][self.ID_POS])
                else:  # Exploration, update group 3
                    while True:
                        pos_new = self.ub - (pop[idx][self.ID_POS] - self.lb) * random()
                        schedule = matrix_to_schedule(self.problem, pos_new.astype(int))
                        if schedule.is_valid():
                            fit_new = self.Fit.fitness(schedule)
                            break
                    while True:
                        rd_id = randint(0, len(g_best) - 1)
                        pos_new_oppo = self.lb + self.ub - g_best[rd_id][self.ID_POS] + random() * (g_best[rd_id][self.ID_POS] - pos_new)
                        schedule = matrix_to_schedule(self.problem, pos_new.astype(int))
                        if schedule.is_valid():
                            fit_new_oppo = self.Fit.fitness(schedule)
                            break
                    if self.dominated(fit_new_oppo, fit_new):
                        temp = pos_new_oppo
                    else:
                        temp = pos_new

                x_new = self.amend_position_random(temp)
                schedule = matrix_to_schedule(self.problem, x_new.astype(int))
                
                if schedule.is_valid():
                    fit_new = self.Fit.fitness(schedule)
                    pop[idx][self.ID_POS] = x_new
                    pop[idx][self.ID_FIT] = fit_new
                    # Update current position, current velocity and compare with past position, past fitness (local best)
                    if Config.METRICS in Config.METRICS_MAX:
                        is_nondominated = True
                        for j in range(len(pop[idx][self.ID_LOCAL_FIT])):
                            if self.is_better(pop[idx][self.ID_LOCAL_FIT][j], fit_new):
                                is_nondominated = False
                                
                        if is_nondominated:
                            for j in reversed(range(len(pop[idx][self.ID_LOCAL_FIT]))):
                                if self.is_better(fit_new, pop[idx][self.ID_LOCAL_FIT][j]):
                                    pop[idx][self.ID_LOCAL_FIT].pop(j)
                                    pop[idx][self.ID_LOCAL_POS].pop(j)
                            pop[idx][self.ID_LOCAL_POS].append(x_new)
                            pop[idx][self.ID_LOCAL_FIT].append(fit_new)
                    else:
                        is_nondominated = True
                        for j in range(len(pop[idx][self.ID_LOCAL_FIT])):
                            if self.is_better(pop[idx][self.ID_LOCAL_FIT][j], fit_new):
                                is_nondominated = False
                                
                        if is_nondominated:
                            for j in reversed(range(len(pop[idx][self.ID_LOCAL_FIT]))):
                                if self.is_better(fit_new, pop[idx][self.ID_LOCAL_FIT][j]):
                                    pop[idx][self.ID_LOCAL_FIT].pop(j)
                                    pop[idx][self.ID_LOCAL_POS].pop(j)
                            pop[idx][self.ID_LOCAL_POS].append(x_new)
                            pop[idx][self.ID_LOCAL_FIT].append(fit_new)
                    break
                
        if fe_mode is None:
            return pop
        else:
            counter = self.pop_size  # pop_new + pop_mutation operations
            return pop, counter


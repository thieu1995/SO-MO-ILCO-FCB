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
from numpy import ceil, sqrt, array, mean, cos
from utils.schedule_util import matrix_to_schedule


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

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # epoch: current chance, self.epoch: number of chances
        wf = 0.5 + 0.5 * (epoch / self.epoch) ** 2   # weight factor
        a = 1.0 - cos(0) * (1.0 / cos(1 - (epoch + 1) / self.epoch))
        coef = 2 - 2 * epoch / (self.epoch - 1)  # linearly decreased from 2 to 0

        for i in range(0, self.pop_size):
            while True:
                rand_number = random()

                if rand_number < 0.1:  # Update using Eq. 1, update from n best position
                    temp = array([random() * pop[j][self.ID_POS] for j in range(0, self.n_sqrt)])
                    temp = mean(temp, axis=0)
                elif 0.1 <= rand_number <= 0.75:  # Exploitation, update group 2
                    if random() < 0.5:
                        if i != 0:
                            better_diff = a * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS])
                        else:
                            better_diff = a * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                        best_diff = (1 - a) * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                        temp = wf * pop[i][self.ID_POS] + uniform() * better_diff + uniform() * best_diff
                    else:
                        # temp = g_best[self.ID_POS] + coef * uniform() * (g_best[self.ID_POS] - wf * pop[i][self.ID_POS])
                        temp = a * g_best[self.ID_POS] + coef * self.get_step_levy_flight(beta=0.5, step=0.01) * \
                               (g_best[self.ID_POS] - uniform() * pop[i][self.ID_POS])
                else:  # Exploration, update group 3
                    while True:
                        pos_new = self.ub - (pop[i][self.ID_POS] - self.lb) * random()
                        schedule = matrix_to_schedule(self.problem, pos_new.astype(int))
                        if schedule.is_valid():
                            fit_new = self.Fit.fitness(schedule)
                            break
                    while True:
                        pos_new_oppo = self.lb + self.ub - g_best[self.ID_POS] + random() * (g_best[self.ID_POS] - pos_new)
                        schedule = matrix_to_schedule(self.problem, pos_new.astype(int))
                        if schedule.is_valid():
                            fit_new_oppo = self.Fit.fitness(schedule)
                            break
                    if fit_new_oppo < fit_new:
                        temp = pos_new_oppo
                    else:
                        temp = pos_new

                child = self.amend_position_random(temp)
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    break
            pop[i] = [child, fit]

        # ## Sort the updated population based on fitness
        # if Config.METRICS in Config.METRICS_MAX:
        #     pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
        # else:
        #     pop = sorted(pop, key=lambda item: item[self.ID_FIT])
        #
        # pop_s1, pop_s2 = pop[:self.n_half], pop[self.n_half:]
        #
        # ## Mutation scheme
        # for i in range(0, self.n_half):
        #     while True:
        #         child = pop_s1[i][self.ID_POS] * (1 + normal(0, 1, self.problem["shape"]))
        #         child = self.amend_position_random(child)
        #         schedule = matrix_to_schedule(self.problem, child)
        #         if schedule.is_valid():
        #             fit = self.Fit.fitness(schedule)
        #             break
        #     pop_s1[i] = [child, fit]
        #
        # ## Search Mechanism
        # pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        # pos_s1_mean = mean(pos_s1_list, axis=0)
        # for i in range(0, self.n_half):
        #     while True:
        #         child = (g_best[self.ID_POS] - pos_s1_mean) - random() * (self.lb + random() * (self.ub - self.lb))
        #         child = self.amend_position_random(child)
        #         schedule = matrix_to_schedule(self.problem, child)
        #         if schedule.is_valid():
        #             fit = self.Fit.fitness(schedule)
        #             break
        #     pop_s2[i] = [child, fit]
        #
        # ## Construct a new population
        # pop = pop_s1 + pop_s2

        if fe_mode is None:
            return pop
        else:
            counter = self.pop_size  # pop_new + pop_mutation operations
            return pop, counter

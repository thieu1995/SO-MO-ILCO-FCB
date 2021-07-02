#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:11, 28/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from numpy import where, sum, any, mean, array, clip, ones, abs, cos
from numpy.random import uniform, choice, normal, randint, random, rand
from scipy.stats import cauchy
from copy import deepcopy
from config import Config
from optimizer.root import Root
from utils.schedule_util import matrix_to_schedule


class SHADE(Root):
    """
        The original version of: Success-History Adaptation Differential Evolution (SHADE)
        Link:
            Success-History Based Parameter Adaptation for Differential Evolution
    """

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"miu_f": 0.5, "miu_cr": 0.5}
        self.miu_f = paras["miu_f"]  # the initial f, default = 0.5
        self.miu_cr = paras["miu_cr"]  # the initial cr, default = 0.5

    ### Survivor Selection
    def weighted_lehmer_mean(self, list_objects, list_weights):
        up = list_weights * list_objects ** 2
        down = list_weights * list_objects
        return sum(up) / sum(down)

    def self_evolve(self, pop, g_best, g_best_list, miu_cr, miu_f, archive_pop, k, time_bound_start):

        for epoch in range(self.epoch):
            time_epoch_start = time()

            list_f = list()
            list_cr = list()
            list_f_index = list()
            list_cr_index = list()

            list_f_new = ones(self.pop_size)
            list_cr_new = ones(self.pop_size)
            pop_new = deepcopy(pop)  # Save all new elements --> Use to update the list_cr and list_f
            pop_old = deepcopy(pop)  # Save all old elements --> Use to update cr value
            sorted_pop = sorted(pop, key=lambda x: x[self.ID_FIT])
            for i in range(0, self.pop_size):
                ## Calculate adaptive parameter cr and f
                idx_rand = randint(0, self.pop_size)
                cr = normal(miu_cr[idx_rand], 0.1)
                cr = clip(cr, 0, 1)
                while True:
                    f = cauchy.rvs(miu_f[idx_rand], 0.1)
                    if f < 0:
                        continue
                    elif f > 1:
                        f = 1
                    break
                list_cr_new[i] = cr
                list_f_new[i] = f
                p = uniform(2 / self.pop_size, 0.2)
                top = int(self.pop_size * p)
                x_best = sorted_pop[randint(0, top)]
                x_r1 = pop[choice(list(set(range(0, self.pop_size)) - {i}))]
                new_pop = pop + archive_pop
                while True:
                    while True:
                        x_r2 = new_pop[randint(0, len(new_pop))]
                        if any(x_r2[self.ID_POS] - x_r1[self.ID_POS]) and any(x_r2[self.ID_POS] - pop[i][self.ID_POS]):
                            break
                    x_new = pop[i][self.ID_POS] + f * (x_best[self.ID_POS] - pop[i][self.ID_POS]) + f * (x_r1[self.ID_POS] - x_r2[self.ID_POS])
                    pos_new = where(uniform(0, 1, self.problem["shape"]) < cr, x_new, pop[i][self.ID_POS])
                    j_rand = randint(0, self.problem["shape"])
                    pos_new[j_rand]= x_new[j_rand]
                    pos_new = self.amend_position_random(pos_new)
                    schedule = matrix_to_schedule(self.problem, pos_new.astype(int))
                    if schedule.is_valid():
                        fit_new = self.Fit.fitness(schedule)
                        break
                pop_new[i] = [pos_new, fit_new]

            for i in range(0, self.pop_size):
                if pop_new[i][self.ID_FIT] < pop[i][self.ID_FIT]:
                    list_cr.append(list_cr_new[i])
                    list_f.append(list_f_new[i])
                    list_f_index.append(i)
                    list_cr_index.append(i)
                    pop[i] = pop_new[i]
                    archive_pop.append(deepcopy(pop[i]))

            # Randomly remove solution
            temp = len(archive_pop) - self.pop_size
            if temp > 0:
                idx_list = choice(range(0, len(archive_pop)), len(archive_pop) - self.pop_size, replace=False)
                archive_pop_new = []
                for idx, solution in enumerate(archive_pop):
                    if idx not in idx_list:
                        archive_pop_new.append(solution)
                archive_pop = deepcopy(archive_pop_new)

            # Update miu_cr and miu_f
            if len(list_f) != 0 and len(list_cr) != 0:
                # Eq.13, 14, 10
                list_fit_old = ones(len(list_cr_index))
                list_fit_new = ones(len(list_cr_index))
                idx_increase = 0
                for i in range(0, self.pop_size):
                    if i in list_cr_index:
                        list_fit_old[idx_increase] = pop_old[i][self.ID_FIT]
                        list_fit_new[idx_increase] = pop_new[i][self.ID_FIT]
                        idx_increase += 1
                list_weights = abs(list_fit_new - list_fit_old) / sum(abs(list_fit_new - list_fit_old))
                miu_cr[k] = sum(list_weights * array(list_cr))
                miu_f[k] = self.weighted_lehmer_mean(array(list_f), list_weights)
                k += 1
                if k >= self.pop_size:
                    k = 0
            g_best, current_best = self.update_g_best_get_current_best(pop, g_best)
            g_best_list.append(g_best[self.ID_FIT])
            time_epoch_end = time() - time_epoch_start
            break_loop = self.check_break_loop(epoch + 1, current_best, g_best, time_epoch_end, time_bound_start)
            if break_loop:
                break

        return g_best, g_best_list

    def train(self):
        time_total = time()
        time_bound_start = time()
        self.check_objective()
        self.check_log()

        ## Initialization
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_g_best(pop)
        g_best_list = [g_best[self.ID_FIT]]
        miu_cr = self.miu_cr * ones(self.pop_size)
        miu_f = self.miu_f * ones(self.pop_size)
        archive_pop = list()
        k = 0

        if Config.MODE == 'epoch':
            g_best, g_best_list = self.self_evolve(pop, g_best, g_best_list, miu_cr, miu_f, archive_pop, k, time_bound_start)
            time_total = time() - time_total
            return g_best[0], g_best[1], array(g_best_list), time_total

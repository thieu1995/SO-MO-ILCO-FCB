#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:32, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from config import Config
from optimizer.root import Root
from numpy.random import uniform, random, choice
from numpy import exp, cos, pi, array
from utils.schedule_util import matrix_to_schedule


class BaseWOA(Root):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"p": 0.5, "b": 1.0}
        self.p = paras["p"]
        self.b = paras["b"]

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        a = 2 - 2 * epoch / (self.epoch - 1)  # linearly decreased from 2 to 0
        for i in range(self.pop_size):
            r = random()
            A = 2 * a * r - a
            C = 2 * r
            l = uniform(-1, 1)

            if uniform() < self.p:
                if abs(A) < 1:
                    while True:
                        child = g_best[self.ID_POS] - A * abs(C * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        child = self.amend_position_random(child)
                        schedule = matrix_to_schedule(self.problem, child.astype(int))
                        if schedule.is_valid():
                            fitness = self.Fit.fitness(schedule)
                            break
                else:
                    while True:
                        id_rand = choice(list(set(range(0, self.pop_size)) - {i}))  # select random 1 position in pop
                        child = pop[id_rand][self.ID_POS] - A * abs(C * pop[id_rand][self.ID_POS] - pop[i][self.ID_POS])
                        child = self.amend_position_random(child)
                        schedule = matrix_to_schedule(self.problem, child.astype(int))
                        if schedule.is_valid():
                            fitness = self.Fit.fitness(schedule)
                            break
            else:
                while True:
                    D1 = abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                    child = D1 * exp(self.b * l) * cos(2 * pi * l) + g_best[self.ID_POS]
                    child = self.amend_position_random(child)
                    schedule = matrix_to_schedule(self.problem, child.astype(int))
                    if schedule.is_valid():
                        fitness = self.Fit.fitness(schedule)
                        break
            pop[i] = [child, fitness]

        if fe_mode is None:
            return pop
        else:
            counter = 2 * self.pop_size  # pop_new + pop_mutation operations
            return pop, counter


class HI_WOA(Root):

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"feedback_max": 10}
        self.feedback_max = paras["feedback_max"]
        self.n_changes = int(pop_size/2)

    def self_evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None, feedback_count=None):
        a = 2 + 2 * cos(pi / 2 * (1 + epoch / self.epoch))  # Eq. 8
        w = 0.5 + 0.5 * (epoch / self.epoch) ** 2  # Eq. 9
        p = 0.5
        b = 1

        for i in range(self.pop_size):
            r = random()
            A = 2 * a * r - a
            C = 2 * r
            l = uniform(-1, 1)

            if uniform() < p:
                if abs(A) < 1:
                    while True:
                        child = w * g_best[self.ID_POS] - A * abs(C * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        child = self.amend_position_random(child)
                        schedule = matrix_to_schedule(self.problem, child.astype(int))
                        if schedule.is_valid():
                            fitness = self.Fit.fitness(schedule)
                            break
                else:
                    while True:
                        id_rand = choice(list(set(range(0, self.pop_size)) - {i}))  # select random 1 position in pop
                        child = pop[id_rand][self.ID_POS] - A * abs(C * pop[id_rand][self.ID_POS] - pop[i][self.ID_POS])
                        child = self.amend_position_random(child)
                        schedule = matrix_to_schedule(self.problem, child.astype(int))
                        if schedule.is_valid():
                            fitness = self.Fit.fitness(schedule)
                            break
            else:
                while True:
                    D1 = abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                    child = w * g_best[self.ID_POS] + exp(b * l) * cos(2 * pi * l) * D1
                    child = self.amend_position_random(child)
                    schedule = matrix_to_schedule(self.problem, child.astype(int))
                    if schedule.is_valid():
                        fitness = self.Fit.fitness(schedule)
                        break
            pop[i] = [child, fitness]

        ## Feedback Mechanism
        current_best = self.get_g_best(pop)
        if current_best[self.ID_FIT] == g_best[self.ID_FIT]:
            feedback_count += 1
        else:
            feedback_count = 0
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = current_best

        if feedback_count >= self.feedback_max:

            idx_list = choice(range(0, self.pop_size), self.n_changes, replace=False)
            pop_new = [self.create_solution() for _ in range(0, self.n_changes)]
            for idx_counter, idx in enumerate(idx_list):
                pop[idx] = pop_new[idx_counter]

        if fe_mode is None:
            return pop, g_best, current_best
        else:
            counter = 2 * self.pop_size  # pop_new + pop_mutation operations
            if feedback_count >= self.feedback_max:
                counter = 2 * self.pop_size + self.n_changes
            return pop, g_best, current_best, counter

    def train(self):
        time_total = time()
        time_bound_start = time()
        self.check_objective()
        self.check_log()

        ## Initialization
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_g_best(pop)
        g_best_list = [g_best[self.ID_FIT]]
        feedback_count = 0

        if Config.MODE == 'epoch':
            for epoch in range(self.epoch):
                time_epoch_start = time()
                pop, g_best, current_best = self.self_evolve(pop, None, epoch, g_best, feedback_count)
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
                pop, g_best, current_best, counter = self.self_evolve(pop, Config.MODE, None, g_best, feedback_count)
                g_best_list.append(g_best[self.ID_FIT])
                fe_counter += counter
                time_fe_end = time() - time_fe_start
                break_loop = self.check_break_loop(fe_counter, current_best, g_best, time_fe_end, time_bound_start)
                if break_loop:
                    break
            time_total = time() - time_total
            return g_best[0], g_best[1], array(g_best_list), time_total
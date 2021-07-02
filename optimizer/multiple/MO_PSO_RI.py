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
from numpy.random import uniform
import random
from utils.schedule_util import matrix_to_schedule
from uuid import uuid4


class BaseMOPSORI(Root3):
    """
    MOPSORI: Multi-objective particle swarm optimization with random immigrants
    """
    ID_IDX = 0
    ID_POS = 1  # Current position
    ID_FIT = 2  # Current fitness
    ID_VEL = 3  # Current velocity
    ID_LOCAL_POS = 4  # Personal best location
    ID_LOCAL_FIT = 5  # Personal best fitness

    def __init__(self, problem=None, pop_size=10, epoch=2, func_eval=100000, lb=None, ub=None, verbose=True, paras=None):
        super().__init__(problem, pop_size, epoch, func_eval, lb, ub, verbose)
        if paras is None:
            paras = {"w_min": 0.4, "w_max": 0.9, "c_local": 1.2, "c_global": 1.2, "decay_rate": 0.5}
        self.w_min = paras["w_min"]
        self.w_max = paras["w_max"]
        self.c_local = paras["c_local"]
        self.c_global = paras["c_global"]
        self.decay_rate = paras["decay_rate"]

    def create_solution(self):
        # print('start')
        while True:
            pos = uniform(self.lb, self.ub, self.problem["shape"])
            schedule = matrix_to_schedule(self.problem, pos.astype(int))
            if schedule.is_valid():
                fitness = self.Fit.fitness(schedule)
                vel = uniform(self.lb, self.ub, self.problem["shape"])
                break
        idx = uuid4().hex
        # print(idx)
        return [idx, pos, fitness, vel, [pos], [fitness]]
        # [solution, fit, velocity, local_solution, local_fitness]

    def evolve(self, pop=None, fe_mode=None, epoch=None, g_best=None):
        # Update weight after each move count  (weight down)
        w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
        imigProb = (1 - (epoch) / (self.epoch)) ** (1 / self.decay_rate)

        for i in range(self.pop_size):
            idx = list(pop.keys())[i]
            if random.random() < imigProb:
                indv = self.create_solution()
                indv[self.ID_IDX] = idx
                pop[idx] = indv
                continue

            while True:
                v_new = w * pop[idx][self.ID_VEL]
                n_locals = len(pop[idx][self.ID_LOCAL_POS])
                n_globals = len(g_best)
                for j in range(n_locals):
                    v_new += 1 / n_locals * self.c_local * uniform() * \
                             (pop[idx][self.ID_LOCAL_POS][j] - pop[idx][self.ID_POS])

                for j in range(n_globals):
                    v_new += 1 / n_globals * self.c_global * uniform() * (g_best[j][self.ID_POS] - pop[idx][self.ID_POS])

                x_new = pop[idx][self.ID_POS] + v_new  # Xi(new) = Xi(old) + Vi(new) * deltaT (deltaT = 1)
                x_new = self.amend_position_random(x_new)
                schedule = matrix_to_schedule(self.problem, x_new.astype(int))
                if schedule.is_valid():
                    fit_new = self.Fit.fitness(schedule)
                    pop[idx][self.ID_POS] = x_new
                    pop[idx][self.ID_FIT] = fit_new
                    pop[idx][self.ID_VEL] = v_new

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

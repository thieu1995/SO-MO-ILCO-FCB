from copy import deepcopy
from optimizer.root import Root
from numpy.random import uniform, normal, random
from numpy import ceil, sqrt, array, mean, cos, std, zeros,  subtract, sum
import numpy as np
from utils.schedule_util import matrix_to_schedule
from random import choice, sample, randint
import math
from bisect import bisect_left


class ILCO_r1(Root):
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
        n_l = self.n_sqrt
        master = sum(array([pop[j][self.ID_POS] for j in range(K)]), axis=0)
        # prob = [(self.pop_size - i) / (self.pop_size + 1) for i in range(self.pop_size)]
        # prob = array(prob / sum(prob))
        # for i in range(1, self.pop_size):
        #     prob[i] += prob[i - 1]
        for i in range(1, self.pop_size):
            while True:
                rand_number = random()
                if rand_number < 0.9:
                    d = normal(0, 0.1)
                    teacher = np.argmin([sum(np.absolute(subtract(pop[i][self.ID_POS], pop[j][self.ID_POS]))) for j in range (K) if j != i]) 
                    new_pop[i][self.ID_POS] = pop[i][self.ID_POS] +\
                        d * (pop[teacher][self.ID_POS] - pop[i][self.ID_POS])  \
                            + normal(0, 0.01, self.problem["shape"])
                else:
                    # pr = random()
                    # _pos = bisect_left(prob, pr)
                    _pos  = randint(0, self.pop_size - 1)
                    friend = new_pop[_pos][self.ID_POS]
                    new_pop[i][self.ID_POS] = wf * friend + (1 - wf) * new_pop[i][self.ID_POS] \
                        + normal(0, 0.01, self.problem["shape"])
                        
                child = self.amend_position_random(new_pop[i][self.ID_POS])
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    new_pop[i][self.ID_POS] = child
                    new_pop[i][self.ID_FIT] = fit
                    break
                
            
        ## find current best used in decomposition

        for i in range(self.pop_size):
            if new_pop[i][self.ID_FIT] < pop[i][self.ID_FIT] or random() < 0.1:
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

                current_best[self.ID_POS] = current_best[self.ID_POS] + normal(0, 0.03, self.problem["shape"])

                child = self.amend_position_random(child)
                schedule = matrix_to_schedule(self.problem, child.astype(int))
                if schedule.is_valid():
                    fit = self.Fit.fitness(schedule)
                    if fit < new_pop[i][self.ID_FIT]:
                        new_pop[i][self.ID_POS] = child
                        new_pop[i][self.ID_FIT] = fit
                    break
        ## Update old population
        # pop = self.update_old_population(pop, new_pop)
        
        for i in range(self.pop_size):
            if new_pop[i][self.ID_FIT] < pop[i][self.ID_FIT] or random() < 0.3:
                pop[i] = new_pop[i]
                
        # pop = new_pop
        if fe_mode is None:
            return pop
        else:
            counter = 2 * self.pop_size  # pop_new + pop_mutation operations
            return pop, counter
#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:14, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from utils.dict_util import ToDict
from uuid import uuid4


class Task(ToDict):

    def __init__(self, t_p=0, t_s=0, label=1, sl_max=10.0, cost_max=1.0):
        self.d_p = t_p      # Device processing
        self.d_s = t_s      # Device storage
        self.label = label      # 0: not saving to blockchain (not important), otherwise: save the blockchain
        self.sl_max = sl_max    # Number of seconds
        self.cost_max = cost_max    # Amount of USD
        self.id = uuid4().hex
        self.total_cost = 0
        
    def reset(self):
        self.total_cost = 0

    def __repr__(self):
        return str(self.to_dict())

    def __eq__(self, other):
        return self.id == other.id and self.sl_max == other.sl_max \
            and self.cost_max == other.cost_max

    def __hash__(self):
        return hash(('id', self.id, 'sl_max', self.sl_max))
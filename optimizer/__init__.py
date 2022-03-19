#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:31, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from optimizer.single.GA import BaseGA
from optimizer.single.PSO import BasePSO
from optimizer.single.WOA import BaseWOA
from optimizer.single.EO import BaseEO
from optimizer.single.AEO import BaseAEO
from optimizer.single.SSA import BaseSSA
from optimizer.single.LCO import BaseLCO

from optimizer.single.PSO import CPSO
from optimizer.single.WOA import HI_WOA
from optimizer.single.AEO import I_AEO
# from optimizer.single.LCO import I_LCO
from optimizer.single.LCO import ILCO
from optimizer.single.VNS import VNS
from optimizer.single.DE import SHADE
from optimizer.single.ILCO_r1 import ILCO_r1
from optimizer.single.ILCO_r2 import ILCO_r2
from optimizer.single.ILCO_r3 import ILCO_r3
from optimizer.single.LCO_a1 import ILCO_a1
from optimizer.single.LCO_a2 import ILCO_a2
from optimizer.single.LCO_a3 import ILCO_a3

from optimizer.multiple.NSGA_II import BaseNSGA_II
from optimizer.multiple.MO_IBLA import MO_IBLA
from optimizer.multiple.NSGA_III import BaseNSGA_III
from optimizer.multiple.NSGAII_SDE import BaseNSGAII_SDE
from optimizer.multiple.NS_SSA import BaseNS_SSA
from optimizer.multiple.MO_SSA import BaseMO_SSA
from optimizer.multiple.MO_ALO import BaseMO_ALO
from optimizer.multiple.MO_ILCO import MO_ILCO
from optimizer.multiple.MO_ILCO_2 import MO_ILCO_2
from optimizer.multiple.MO_PSO_RI import BaseMOPSORI
from optimizer.multiple.improved_NSGA_III import Improved_NSGA_III


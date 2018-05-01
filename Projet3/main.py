#!/usr/bin/env python
# from FeuRouge import *
import numpy as np
import pyAgrum.lib.notebook as gnb
from MouseInMaze import MouseInMaze
from MonoBestiole import MonoBestiole
from PeriodicCdM import PeriodicCdM
from CdMConvergence import CdMConvergence
from Oie import Oie
from CdMSampler import CdMSampler
from CollTimeOut import CollTimeOut
from CollOieFinish import CollOieFinish
from CollTempsMoyen import CollTempsMoyen

if __name__ == '__main__':
    # f = FeuRouge()
    # f.distribution_to_vector({"Rouge": 0.7, "Vert": 0.3})

    # f = FeuRouge()
    # a = f.vector_to_distribution(np.array([0, 0.5, 0.5]))
    # print(a)

    # f = FeuRouge()
    # b = f.show_distribution(f.get_initial_distribution())
    # print(b)

    # f = FeuRouge()
    # c= f.get_transition_matrix()
    # print(c)

    # f = FeuRouge()
    # gnb.showDot(f.get_transition_graph().toDot())
    #
    # f = FeuRouge()
    # f.show_transition_graph(gnb)

    # m = MouseInMaze()
    # m.is_ergodic();

    # m = MonoBestiole(6, 0.5, 0.5)
    # cdm = CdMConvergence(m)
    # cdm.point_fixe()
    # m.convergence_M_n()
    # m = MonoBestiole(6, 0.5, 0.5)
    # mouse = MouseInMaze()
    # p = PeriodicCdM()
    # cdm = CdMConvergence(m)
    # cdm.point_fixe()
    # cdm = CdMConvergence(mouse)
    # cdm.point_fixe()
    # cdm = CdMConvergence(p)
    # cdm.point_fixe()


    # p=PeriodicCdM()
    # p.is_ergodic()

    o = Oie(50, 6)
    graph=o.makeGraph()
    print(graph)

    sampler=CdMSampler(o)
    # sampler.add_collector(CollGetDistribution(epsilon=1e-5,pas=20000))
    # sampler.add_collector(CollTimeOut(10)) # time out de 10 seconde
    # sampler.add_collector(CollOieFinish())
    sampler.add_collector(CollTempsMoyen())
    sampler.run(100000)
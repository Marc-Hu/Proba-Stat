#!/usr/bin/env python
# from FeuRouge import *
import numpy as np
import pyAgrum.lib.notebook as gnb
from MouseInMaze import MouseInMaze
from MonoBestiole import MonoBestiole
from PeriodicCdM import PeriodicCdM

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

    m = MonoBestiole(5, 0.5, 0.5)
    m.is_ergodic()

    # p=PeriodicCdM()
    # p.is_ergodic()
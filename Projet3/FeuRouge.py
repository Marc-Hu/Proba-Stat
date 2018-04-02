#!/usr/bin/env python
#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import utils
from CdM import CdM


# feu rouge hérite de CdM
class FeuRouge(CdM):
    def __init__(self):
        self.stateToIndex = { 'Orange': 1, 'Rouge': 0, 'Vert': 2}
        super(FeuRouge, self).__init__()

    def get_states(self):
        return ['Rouge', 'Orange', 'Vert']

    def get_transition_distribution(self, state):
        if state == 'Rouge':
          return {'Rouge': 0.8, 'Vert': 0.2}
        elif state == 'Orange':
          return {'Orange': 0.7, 'Rouge': 0.3}
        elif state == 'Vert':
          return {'Vert': 0.8, 'Orange': 0.2}
        else:
          raise IndexError

    def get_initial_distribution(self):
        return {'Vert': 0.3, 'Rouge': 0.7}


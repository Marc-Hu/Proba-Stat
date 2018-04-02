import numpy as np
from CdM import CdM

class MouseInMaze(CdM):
    def __init__(self):
        super(MouseInMaze, self).__init__()


    def get_states(self):
        return [1, 2, 3, 4, 5, 6]

    def get_transition_distribution(self, state):
        if state == 1:
            return {1 : 0.5, 2 : 0.5}
        elif state == 2:
            return {1 : 0.5, 4 : 0.5}
        elif state == 3:
            return {1 : 0.25, 2 : 0.25, 5 : 0.25, 6 : 0.25}
        elif state == 4:
            return {3 : 1.0}
        elif state == 5:
            return {5 : 1.0}
        elif state == 6:
            return {6 : 1.0}

    def get_initial_distribution(self):
        return {2 : 1.0}

    def distribution_to_vector(self, distribution):
        l = len(self.get_states())
        vector = np.zeros((1, l))
        for k, v in distribution.items():
            vector[0][k-1] = v
        return vector[0]
        

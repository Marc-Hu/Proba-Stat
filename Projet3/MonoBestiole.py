import numpy as np
from CdM import CdM


class MonoBestiole(CdM):
    def __init__(self, nbEtat, proba_droite, proba_gauche):
        self.stateToIndex = {}
        self.nbEtat=nbEtat
        self.p_droite=proba_droite
        self.p_gauche=proba_gauche
        for i in range(nbEtat):
            self.stateToIndex[str(i+1)] = int(i)
        super(MonoBestiole, self).__init__()

    def get_states(self):
        return range(1, self.nbEtat+1)

    def get_transition_distribution(self, state):
        droite = state+1
        gauche = state-1
        if state == 1:
            gauche = 1
        elif state == self.nbEtat:
            droite = self.nbEtat
        return {gauche: self.p_gauche, droite: self.p_droite}

    def get_initial_distribution(self):
        return { '1' : 0.3, '2': 0.1, str(self.nbEtat-1):0.2, str(self.nbEtat):0.4}



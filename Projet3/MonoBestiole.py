import numpy as np
from CdM import CdM
import matplotlib.pyplot as plt
import utils
import random


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
        return list(range(1, self.nbEtat+1))

    def get_transition_distribution(self, state):
        droite = state+1
        gauche = state-1
        if gauche == 0:
            gauche = 1
        elif droite == self.nbEtat+1:
            droite = self.nbEtat
        return {gauche: self.p_gauche, droite : self.p_droite}

    def get_initial_distribution(self):
        return { '1' : 0.3, '2': 0.1, str(self.nbEtat-1):0.2, str(self.nbEtat):0.4}
        # return { str(random.randint(1, len(self.get_states()))) : 1}

    def show_distribution(self, dist):
        # print("distribution : ", distribution)
        distribution = {}
        for key, value in dist.items() :
            distribution[str(key)] = value
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 1)
        ax.set_yticks([])
        ax.set_xticklabels(self.get_states())
        ax.set_xticks(np.arange(0, len(self.get_states()), step=1))
        # print("distribution to vector : ", self.distribution_to_vector(distribution))
        ax.imshow(self.distribution_to_vector(distribution).reshape(1, len(self.get_states())), cmap=utils.ProbaMap)

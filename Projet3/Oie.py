import numpy as np
from CdM import CdM
import matplotlib.pyplot as plt
import utils
import random


class Oie(CdM):
    def __init__(self, nbCase, d):
        self.stateToIndex = {}
        self.nbCase=nbCase
        self.cases = [0]*nbCase
        self.set_trap()
        self.nbface_d = d
        super(Oie, self).__init__()

    def set_trap(self):
        nbGlissage = int(((random.randint(67, 100)/100)*self.nbCase/10)/2)
        nbTremplin = nbGlissage
        nbPuit = int(self.nbCase/10)-nbGlissage*2
        trap = [nbGlissage, nbTremplin, nbPuit]
        for i in range(int(self.nbCase/10)):
            case = random.randint(0, self.nbCase)
            while case >= self.nbCase-1 or  not self.cases[case]==0 or self.cases[case+1]==-1 or case==0:
                case = random.randint(0, self.nbCase)
            trapRandom = random.randint(0, len(trap)-1)
            while trap[trapRandom]==0:
                trapRandom = random.randint(0, len(trap) - 1)
            if trapRandom==0:
                self.cases[case] = -1
            elif trapRandom==1:
                self.cases[case] = 1
            else:
                self.cases[case] = -2
            trap[trapRandom]=trap[trapRandom]-1
        # print(self.cases)


    def get_states(self):
        return list(range(1, self.nbCase + 1))

    def get_transition_distribution(self, state):
        if state == self.nbCase :
            return {1 : 1.0}
        if self.cases[state-1] == -1 or self.cases[state-1] == 1 :
            return {state+self.cases[state-1] : 1.0}
        transition = {}
        j=0
        for i in range(1, self.nbface_d+1) :
            if state+i>self.nbCase:
                j=j-2
            if state+i+j in transition :
                transition[state + i + j] = transition[state + i + j] + 1 / self.nbface_d
            else :
                transition[state+i+j] = 1/self.nbface_d
        return transition

    def get_initial_distribution(self):
        return { 1 : 1.0}

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

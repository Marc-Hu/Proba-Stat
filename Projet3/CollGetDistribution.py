from Collector import Collector
import numpy as np


class CollGetDistribution(Collector):
    def __init__(self, epsilon, pas):
        """
        Constructeur
        :param epsilon:
        :param pas:
        """
        self.epsilon = epsilon
        self.pas = pas
        self.dico_etat_visit = {} # Le dico des états visité
        self.distribution = {} # La distribution à n-1
        self.error = 0

    def initialize(self, cdm, max_iter):
        """
        Méthode qui va initialiser la distribution initiale dans self.distribution
        :param cdm:
        :param max_iter:
        :return:
        """
        self.distribution=cdm.get_initial_distribution()
        # print(self.distribution)

    def receive(self, cdm, iter, state):
        # print(iter, state)
        # Initialisation du dico du nombre d'états visités
        if state in self.dico_etat_visit:
            self.dico_etat_visit[state] += 1
        else:
            self.dico_etat_visit[state] = 1
        distri_courante = {} # La distribution courante
        # print(self.dico_etat_visit)
        for (k, v) in self.dico_etat_visit.items():
            distri_courante[k] = v / iter

        vector_old = cdm.distribution_to_vector(self.distribution)
        vector_current = cdm.distribution_to_vector(distri_courante)
        # print(vector_current, vector_old)
        diff = np.subtract(np.array(vector_current), np.array(vector_old))  # différence entre distri_courante et distribution
        # print("diff : ", diff)
        self.error = np.amax(np.abs(np.array(diff)))  # On stock dans self.error la valeur max de la diff
        self.distribution = distri_courante  # Sinon distribution=distri_courante
        # print(iter, self.pas)
        if iter % self.pas == 0:
            cdm.show_distribution(self.distribution)
        # print(self.error)
        # if self.error<self.epsilon :
        #     return True
        return False

    def finalize(self, cdm, iteration):
        pass

    def get_results(self, cdm):
        return {"erreur": self.error, "proba": self.distribution}
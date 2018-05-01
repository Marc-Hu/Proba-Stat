# -*- coding: utf-8 -*-
import time

from Collector import Collector


class CollTempsMoyen(Collector):
    """
    Classe qui permet d'enregistrer le temps moyen d'une partie d'Oie
    """
    def __init__(self):
        self.temps=[]
        self.start=0
        self.nb_retour=0

    def initialize(self, cdm, max_iter):
        self.start=time.clock()

    def receive(self, cdm, iter, state):
        if state == len(cdm.get_states()) : # Si on atteint le dernier état
            self.temps.append(time.clock() - self.start) # On enregistre le temps
            self.start=time.clock() # On remet à zero le chrono
            self.nb_retour=self.nb_retour+1 # On incrément le nombre de retour
        return False

    def finalize(self, cdm, iteration):
        pass

    def get_results(self, cdm):
        return {"Nombre de retour vers 1 ": self.nb_retour, "Temps Moyen ": sum(self.temps)/len(self.temps)}

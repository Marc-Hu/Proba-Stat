# -*- coding: utf-8 -*-
import time

from Collector import Collector


class CollOieFinish(Collector):
    def __init__(self):
        self.iteration=0

    def initialize(self, cdm, max_iter):
        pass

    def receive(self, cdm, iter, state):
        self.iteration=self.iteration+1
        if state == len(cdm.get_states()) :
            return True
        return False

    def finalize(self, cdm, iteration):
        pass

    def get_results(self, cdm):
        return {"Gagné à l'itération": self.iteration}

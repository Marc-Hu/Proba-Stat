from Collector import Collector


class CollGetDistribution(Collector):
    def __init__(self, epsilon, pas):
        self.epsilon = epsilon
        self.pas = pas
        self.max_iter=0

    def initialize(self, cdm, max_iter):
        # print("run({}): ".format(max_iter), end="", flush=True)
        # print(cdm, max_iter)
        self.max_iter=max_iter
        print(cdm.get_initial_distribution())

    def receive(self, cdm, iter, state):
        # print(iter, state)
        # print(cdm.get_transition_distribution(int(state)))
        print(state)
        cdm.show_distribution(cdm.get_transition_distribution(int(state)))

    def finalize(self, cdm, iteration):
        print("finalize")

    def get_results(self, cdm):
        return {'proba' : {1:0.1}}
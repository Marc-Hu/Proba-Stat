# -*- coding: utf-8 -*-
import random
import numpy as np

class CdMSampler:
  def __init__(self, cdm):
    self.cdm = cdm
    self.collectors = []
    # simulation effectuée
    self.done = False

  def add_collector(self, collector):
    """
    Ajoute un nouveau collector (instance d'une classe qui hérite de Collector)
    :param collector: le nouveau collector
    """
    self.collectors.append(collector)

  def notify_initialize(self, nbr_iter):
    """
    Notifie tout collector qu'une simulation de taille maximum nbr_iter va commencer
    :param nbr_iter: nombre maximum d'itérations de la simulation
    """
    for collector in self.collectors:
      collector.initialize(self.cdm, nbr_iter)

  def notify_receive(self, iteration, state):
    """
    Notifie à tout collector le passage de la simulation à l'état `state` dans l'itération `iteration`.

    :param iteration: indice de l'itération courante
    :param state: état atteint dans cette itération
    :return: True : si un collector a indiqué qu'il voulait que la simulation s'arrête
             False : si tout collector est OK pour continuer la simulation
    """
    res = False
    for collector in self.collectors:
      if collector.receive(self.cdm, iteration, state):
        res = True
    return res

  def notify_finalize(self, nbr_iteration):
    """
    Notifie tout collector que la simulation se termine à l'itération `nbr_iteration`
    :param nbr_iteration: l'indice de la dernière itération
    """
    for collector in self.collectors:
      collector.finalize(self.cdm, nbr_iteration)

  def collect_results(self, nbr_iteration):
    """
    Intégre dans un seul dictionnaire l'ensemble des résutalts des collectors
    :param nbr_iteration: l'indice de la dernière itération
    :return: le dictionnaire des résultats
    """
    if not self.done:
      return None

    res = {'nbr_iterations': nbr_iteration}
    for collector in self.collectors:
      r = collector.get_results(self.cdm)
      if r is not None:
        for key, val in r.items():
          # plusieurs fois la même clé (cf. CollSingleStateCounter) => on crée un dictionnaire avec toutes les
          # valeurs engrangées
          if key in res:
            if type(res[key]) is not dict:
              res[key] = {0: res[key]}

            if type(val) is dict:
              res[key].update(val)
            else:
              res[key][1 + max(res[key].keys())] = val
          else:
            res[key] = val
    return res

  @staticmethod
  def draw_from_distribution(distribution):
    """
    tire aléatoirement un état suivant la distribution
    :param distribution: la distribution de tirage
    :return: l'état tiré aléatoirement suivant la distribution
    """
    f = random.uniform(0.0, 1.0)
    for k in distribution.keys():
    	if f < distribution[k]:
    		# Il faut un truc ici sinon c la ...
    		return k

  def run(self, max_iter):
    """
    Effectue une simulation complète, de taille maximum max_iter
    :param max_iter: nombre maximum d'itérations
    :return: le dictionnaire des résultats collectés par les collector
    """
    self.notify_initialize(max_iter)
    for i in range(1, max_iter+1):
    	state = self.draw_from_distribution(self.cdm.get_initial_distribution())
    	r = self.notify_receive(i, state)
    	if r:
    		break
    	if i == max_iter:
    		self.notify_finalize(max_iter)
    self.done = True
    return self.collect_results(max_iter)
    	

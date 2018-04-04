# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors

c = np.zeros((100, 4))
c[:, -1] = np.linspace(0, 1, 100)  # gradient de transparence
c[:, 0] = 0.3  # rouge
c[:, 1] = 0.5  # vert
c[:, 2] = 0.5  # bleu
# ProbaMap est une matplotlib.colormap qu'on utilisera pour afficher des valeurs de probabilités (de 0 à 1).
ProbaMap = colors.ListedColormap(c)


def show_matrix(matrix):
  """
  :warning: ne devrait pas être surchargé
  présente la matrice de transition
  :param matrix: np.array qui devrait être une matrice stochasique
  """
  plt.matshow(matrix, cmap=ProbaMap)
  plt.grid(False)
  plt.show()


def pgcd(a, b): 
  """pgcd(a,b): calcul du 'Plus Grand Commun Diviseur' entre les 2 nombres entiers a et b"""
  while b != 0:
    a, b = b, a % b

  return a


def dfs(g):
  """
  Depth-first search dans le graphe g.
  Cette fonction n'est qu'un prototype de code : elle ne fait rien à part le parcours en profondeur d'abord dans
  pyAgrum.DiGraph.
  :param g:
  :return:
  """
  mark = dict()

  def _dfs(node):
    if node in mark:  # déja visité
      return

    mark[node] = True
    for fils in g.children(node):
      _dfs(fils)

  for i in g.ids():
    if i not in mark:
      _dfs(i)

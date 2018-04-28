#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
# import matplotlib.pyplot as plt
import tarjan
from decimal import Decimal
import math

import utils


class CdM():
    """
    Class virtuelle représentant une Chaîne de Markov
    """

    def __init__(self):
        """
        Constructeur. En particulier, initalise le dictionaire stateToIndex

        :warning: doit être appelé en fin de __init__ des classes filles
        avec ` super().__init__()`
        """
        self.stateToIndex
        pass

    def get_states(self):
        """
        :return: un ensemble d'états énumérable (list, n-uple, etc.)
        """
        raise NotImplementedError

    def get_transition_distribution(self, state):
        """
        :param state: état initial
        :return: un dictionnaire {etat:proba} représentant l'ensemble des états atteignables à partir de state et leurs
        probabilités
        """
        raise NotImplementedError

    def get_initial_distribution(self):
        """
        :return: un dictionnaire représentant la distribution à t=0 {etat:proba}
        """
        raise NotImplementedError

    def __len__(self):
        """
        permet d'utiliser len(CdM) pour avoir le nombre d'état d'un CdM

        :warning: peut être surchargée
        :return: le nombre d'état
        """
        return len(self.get_states())

    def show_transition_matrix(self):
        """
        Affiche la matrice de transition

        :return:
        """
        utils.show_matrix(self.get_transition_matrix())

    def distribution_to_vector(self, distribution):
        """
        Convertit la distribution en vecteur
        :param distribution: Distribution (format dict) à convertir
        :return: Le vecteur de la distribution
        """
        l = len(self.get_states())
        vector = np.zeros((1, l))
        for k, v in distribution.items():
            index = self.stateToIndex.get(k)
            vector[0][index] = v
        return vector[0]

    def vector_to_distribution(self, vector):
        """
        Convertit un vecteur en distribution
        :param vector: Vecteur (array) en distribution (dict)
        :return: La distribution du vecteur
        """
        list = {}
        for i in range(len(vector)):
            if not vector[i] == 0.:
                list[self.get_states()[i]] = vector[i]
        return list

    def show_distribution(self, distribution):
        """
        Permet de représenter une distribution
        :param distribution: Distribution à représenter
        :return:
        """
        res = [0] * self.__len__()
        for i in range(len(self.get_states())):
            if self.get_states()[i] in distribution:
                res[i] = distribution[self.get_states()[i]]
        return res

    def get_transition_matrix(self):
        """
        Permet de construire un numpy.array  représentant la matrice  du MdP
        :return: le numpy.array qui représente la matrice MdP
        """
        size = len(self.get_states())
        state = self.get_states()
        array = np.zeros((size, size))
        for i in range(array.shape[0]):
            distribution = self.get_transition_distribution(state[i])
            for j in range(array.shape[1]):
                if state[j] in distribution:
                    array[i][j] = distribution[state[j]]
        return array

    def get_transition_graph(self):
        """
        Crée un gum.DiGraph représentant la structure du graphe de transition
        :return:
        """
        array = self.get_transition_matrix()
        state = self.get_states()
        g = gum.DiGraph()
        for i in range(len(state)):
            g.addNode()

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] != 0.:
                    g.addArc(i, j)

        return g

    def show_transition_graph(self, gnb):
        """
        Dessine le graphe de transition (avec les paramètres)
        :param gnb: Le module qu'on utilise pour dessiner
        :return:
        """
        array = self.get_transition_matrix()
        res = "digraph {\n"
        state = self.get_states()
        for i in range(len(state)):
            res += str("  " + str(i) + " [label=\"[" + str(i) + "] " + str(state[i]) + "\"];\n")
        res += "\n"
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] != 0.:
                    res += str("  " + str(i) + '->' + str(j) + " [label=" + str(array[i][j]) + "];\n")
        res += "}"
        gnb.showDot(res)

    def get_communication_classes(self):
        """
        Méthode qui retourne les composantes fortement connexes du graphe du CdM
        :return:
        """
        return tarjan.tarjan(self.makeGraph())
        # return self.dfs(graph)

    def makeGraph(self):
        """
        Méthode qui va construire un graphe par rapport à la matrice de transition
        :return:
        """
        graph = {}
        component = []
        for i in range(len(self.get_states())):
            for j in range(len(self.get_transition_matrix()[i].tolist())):
                if not self.get_transition_matrix()[i].tolist()[j] == 0:
                    component.append(self.get_states()[j])
            # print(component)
            graph[self.get_states()[i]] = component
            component = []
        return graph

    def get_absorbing_classes(self):
        """
        Méthode qui permet de connaître les classes absorbante
        :return:
        """
        result = []
        notirreductibleresult = []
        found = False
        graph = self.makeGraph()
        for key, value in graph.items():
            for i in range(len(value)):
                if not value[i] == key:
                    result.append(True)
                    found = True
                    break
            if not found:
                notirreductibleresult.append([key])
            found = False
        if len(result) == len(graph):
            return self.get_communication_classes()
        return notirreductibleresult

    def is_irreducible(self):
        """
        Méthode qui permet de savoir si un graphe est irreductible ou non
        :return:
        """
        if len(self.get_states()) == len(self.get_absorbing_classes()[0]):
            return True
        return False

    def find_all_paths(self, graph, start, end, path=[]):
        """
        Méthode qui va chercher et renvoyer tous les chemins possibles d'un graph
        :param graph: Graphe à parcourir
        :param start: L'état de départ
        :param end: L'état à atteindre
        :param path: Chemin en cours d'exploitation
        :return: Tous les chemins possible entre l'état start et l'état end
        """
        path = path + [start] # Ajout de l'état start dans le path
        if start == end: # Si on est arrivé alors on retourne le path
            return [path]
        if start not in graph: # Si l'état n'appartient pas au graph
            return [] # On retourne un tableau vide
        paths = [] # Variable qui va contenir tous les paths
        for node in graph[start]: # Pour tous les successeur de l'état start
            if node not in path: # Si le successeur n'est pas encore dans le path
                newpaths = self.find_all_paths(graph, node, end, path) # On va faire un appel récursive pour chercher le path du successeur
                for newpath in newpaths: # Pour tous les nouveaux path
                    paths.append(newpath) # On ajoutes ces paths dans paths
        return paths

    def get_periodicity(self):
        """
        Méthode pour connaître la périodicité d'un Cdm
        :return:
        """
        # print(self.makeGraph())
        graph = self.makeGraph()
        # print(graph)
        result = []
        if self.is_irreducible():  # Si c'est irreductible
            for key, value in graph.items():  # Pour chaque état key
                for i in range(len(value)):  # Pour chaque successeurs de la clé
                    # On va chercher tous les chemins possible entre le key et le key
                    paths = [[[key] + y for y in self.find_all_paths(graph, x, key)] for x in graph[key]]
                    length_path = []  # Tableau qui contiendra les longueurs des chemins
                    for j in range(len(paths)):  # Pour tous les chemins trouvés
                        length_path.append(len(paths[j][0]) - 1)  # On ajoute la taille - 1 du chemin j
                    if len(length_path) == 1:  # Si il y a un seul chemin
                        result.append(length_path[0])
                    else:
                        # print("length path", length_path)
                        # On calcule le pgcd des deux première valeurs
                        pgcd_result = utils.pgcd(length_path[0], length_path[1])
                        for k in range(2, len(length_path)): # On boucle si il y a plus de 2
                            pgcd_result = utils.pgcd(pgcd_result, length_path[k])
                        result.append(pgcd_result) # On va ajouter le resultat du pgcd d'un état
            # print(result)
            # On fait le pgcd des deux premier resultat
            pgcd_result = utils.pgcd(result[0], result[1])
            for k in range(2, len(result)): # Et on continue jusqu'a la fin
                pgcd_result = utils.pgcd(pgcd_result, result[k])
            # print(pgcd_result)
            return pgcd_result
        return

    def is_aperiodic(self):
        """
        Méthode qui va vérifier si le Cdm est aprériodique ou non
        :return: Un boolean pour savoir si c'est apériodique
        """
        if self.get_periodicity() == 1:
            return True
        return False

    def is_ergodic(self):
        """
        Méthode qui va regarder si un CdM est ergodique ou non
        :return:
        """
        print(self.is_irreducible())
        if self.is_irreducible() and self.is_aperiodic():  # Si c'est ergodique et aperiodique
            print(self.get_transition_matrix())
            position = self.distribution_to_vector(self.get_initial_distribution())  # Position initiale
            result = [0] * len(self.get_states())
            for i in range(100):  # Nb d'itération
                for j in range(len(position)):  # Pour chaque position
                    res = 0
                    for k in range(len(position)):  # Pour chaque position
                        # print(self.get_transition_matrix()[j][k])
                        # Si la valeur de la transition est différente de 0
                        if not self.get_transition_matrix()[j][k] == 0:
                            # On modifie la valeur de l'état k
                            res = res + position[k] * self.get_transition_matrix()[j][k]
                            # print(j)
                    result[j] = round(res, 4)  # On arrondi le résultat
                # print(position, result)
                # On regarde si sa converge en regardant le précédent tableau avec le nouveau
                if self.check_array_equals(position, result):
                    print(True, position, result)
                    return True
                position = result.copy()
        print(False, position, result)
        return False

    def check_array_equals(self, array1, array2):
        """
        Méthode qui va comparer deux tableaux et inspecter la différence entre
        ces deux tableaux
        :param array1: Premier tableau
        :param array2: Deuxième tableau
        :return: True si la somme des différences entre ces deux tableaux n'est pas trop
        grande
        """
        res = 0.0
        for i in range(len(array1)):
            if array1[i] > array2[i]:
                res = res + array1[i] - array2[i]
            else:
                res = res + array2[i] - array1[i]
        # print(res)
        if res < 0.001:
            return True
        return False
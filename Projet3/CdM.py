#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import matplotlib.pyplot as plt

import utils


class CdM(object):
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
            if not vector[i]==0.:
                list[self.get_states()[i]]=vector[i]
        return list

    def show_distribution(self, distribution):
        """
        Permet de représenter une distribution
        :param distribution: Distribution à représenter
        :return:
        """
        res=[0]*self.__len__()
        for i in range(len(self.get_states())):
            if self.get_states()[i] in distribution :
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
                    array[i][j]=distribution[state[j]]
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
                if array[i][j]!=0. :
                    g.addArc(i, j)

        return g

    def show_transition_graph(self, gnb):
        """
        Dessine le graphe de transition (avec les paramètres)
        :param gnb: Le module qu'on utilise pour dessiner
        :return:
        """
        array = self.get_transition_matrix()
        res="digraph {\n"
        state = self.get_states()
        for i in range(len(state)):
            res += str("  "+str(i)+" [label=\"["+str(i)+"] "+str(state[i])+"\"];\n")
        res += "\n"
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j]!=0. :
                    res += str("  "+str(i)+'->'+str(j)+" [label="+str(array[i][j])+"];\n")
        res+="}"
        print(res)
        gnb.showDot(res)


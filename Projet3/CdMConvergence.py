import numpy as np
import matplotlib.pyplot as plt
import time


class CdMConvergence():

    def __init__(self, cdm):
        self.cdm = cdm # CdM ou on doit appliquer la convergence
        self.erreur_pi = [] # Evolution des erreurs de pi
        self.erreur_M = [] # Evolution des erreurs de M
        self.nbRun = 20 # Nombre de run pour récupérer des données
        self.temps_pi=[] # Liste des temps récupérer pour chaque run de la convergence de pi
        self.temps_M=[] # Liste de temps récupérer pour chaque run de la convergence de M
        self.temps_point_fixe=[]
        self.temps_ergodic=[] # Liste de temps récupérer pour chaque run de la convergence de ergodic (1er méthode)

    def is_ergodic(self):
        """
        Première méthode
        :return:
        """
        if self.cdm.is_irreducible() and self.cdm.is_aperiodic():  # Si c'est ergodique et aperiodique
            # print(self.get_transition_matrix())
            start = time.clock()
            position = self.cdm.distribution_to_vector(self.cdm.get_initial_distribution())  # Position initiale
            result = [0] * len(self.cdm.get_states())
            for i in range(100):  # Nb d'itération
                for j in range(len(position)):  # Pour chaque position
                    res = 0
                    for k in range(len(position)):  # Pour chaque position
                        # print(self.get_transition_matrix()[j][k])
                        # Si la valeur de la transition est différente de 0
                        if not self.cdm.get_transition_matrix()[j][k] == 0:
                            # On modifie la valeur de l'état k
                            res = res + position[k] * self.cdm.get_transition_matrix()[j][k]
                            # print(j)
                    result[j] = round(res, 4)  # On arrondi le résultat
                # print(position, result)
                # On regarde si sa converge en regardant le précédent tableau avec le nouveau
                if self.check_array_equals(position, result, 0.001):
                    # print(True, position, result)
                    self.temps_ergodic.append(time.clock() - start)
                    return True, i, position
                position = result.copy()
            self.temps_ergodic.append(time.clock()-start)

    def convergence_pi_n(self, epsilon):
        """
        Méthode qui va calculer la convergence de pi n et s'arrête lorsque la
        différence entre n et n-1 est assez faible
        :return:
        """
        if not self.cdm.is_ergodic():
            return False
        array = np.zeros((1, len(self.cdm.get_states()))) # pi à n
        array_n_minus_one = array.copy() # pi à n-1
        # On initialise pi(0)
        for key, value in self.cdm.get_initial_distribution().items():
            try:
                array[0][int(key) - 1] = value
            except:
                array[0][self.cdm.get_states().index(key)] = value
        start = time.clock()
        # print(start)
        self.erreur_pi=[]
        for i in range(100): # Itération (la valeur peut être modifié)
            # Multiplication de la matrice array avec la matrice de transition
            array = np.dot(array, self.cdm.get_transition_matrix())
            # On verifie si la différence entre les deux matrices est assez faible (selon epsilon)
            if self.check_array_equals(array[0], array_n_minus_one[0], epsilon):
                # print("Convergence de pi à l'itération : ", i, array)
                self.temps_pi.append(time.clock() - start)
                return True, i, array
                # break
            array_n_minus_one=array.copy()
        self.temps_pi.append(time.clock()-start)
        # print("Pi n'as pas convergé")
        # return array[0]

    def check_array_equals(self, array1, array2, epsilon):
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
        self.erreur_pi.append(res)
        if res < epsilon:
            return True
        return False

    def convergence_M_n(self, epsilon):
        """
        Méthode qui calcul la convergence de Mn
        :return: Booleen si on a réussi à converger, la position à laquelle
        on a convergé, le point
        """
        if not self.cdm.is_ergodic():
            return False
        # print(array.shape)
        array = self.cdm.get_transition_matrix()
        array_n_minus_one = array.copy()
        start = time.clock()
        self.erreur_M=[]
        for i in range(100):
            array=np.dot(array, array)
            # print(array)
            if self.check_matrix_converge(array, array_n_minus_one, epsilon):
                # print("Convergence de M à l'itération : \n", i, array)
                self.temps_M.append(time.clock() - start)
                return True, i, array
                # break
            array_n_minus_one=array.copy()
        self.temps_M.append(time.clock()-start)
        # return False, i, array[0]

    def check_matrix_converge(self, matrix1, matrix2, epsilon):
        """
        Méthode qui va comparer deux matrices
        :param matrix1:
        :param matrix2:
        :return: True si la différence entre les deux matrices est assez faible
        """
        dif = 0.0
        for i in range(matrix1.shape[0]):
            for j in range(matrix1.shape[1]):
                if matrix1[i][j]<matrix2[i][j] :
                    dif = dif + (matrix2[i][j]-matrix1[i][j])
                else :
                    dif = dif + (matrix1[i][j]-matrix2[i][j])
        self.erreur_M.append(dif)
        if dif < epsilon:
            return True
        return False

    def point_fixe(self):
        """
        Méthode qui calcul le point fixe
        :return:
        """
        if not self.cdm.is_ergodic():
            return False
        start = time.clock()
        pi_n = self.convergence_pi_n(0.000001)
        # M_n = self.convergence_M_n(0.000001)
        # print("Vecteur propre de M pour la valeur propre 1 : \n", np.dot(pi_n[2][0], self.cdm.get_transition_matrix()))
        self.temps_point_fixe.append(time.clock()-start)

        # print("Partie Point Fixe modifiée")
        # matrice_transition = self.cdm.get_transition_matrix()
        # valeurs, vecteurs = np.linalg.eig(matrice_transition)
        # # print("Valeurs propres", valeurs)
        # # print("Vecteurs propres: ", vecteurs)
        # position = np.where(valeurs == 1)
        # # print(position)
        # print("Le point fixe: ", vecteurs[position])
        # return vecteurs[position]

    def showErrorBending(self, errorname):
        """
        Méthode qui va afficher la courbe des erreur selon la catégorie qu'on veut
        entré dans le paramètre errorname
        :param errorname: Nom de la courbe d'évolution que l'on veut
        :return: rien
        """
        plt.title("Evolution de l'erreur pour "+errorname)
        plt.xlabel("Itération")
        plt.ylabel("Pourcentage d'erreur")
        if errorname == "pi":
            plt.plot(self.erreur_pi)
        elif errorname == "M" :
            plt.plot(self.erreur_M)
        else :
            plt.plot(self.cdm.get_ergodic_error())
        plt.show()

    def get_temps(self):
        """
        Méthode qui va récupérer les temps d'exécutions des différentes méthode
        :return: Les temps d'exécutions pour les méthodes Pi, M, ergodic et point fixe
        """
        # print(self.temps_pi, self.temps_M)
        self.temps_pi = []
        self.temps_M = []
        self.temps_point_fixe = []
        self.temps_ergodic = []
        for i in range(self.nbRun): # Pour chaque run
            # print(self.temps_ergodic)
            self.is_ergodic()
            self.convergence_pi_n(0.000001)
            self.convergence_M_n(0.000001)
            self.point_fixe()
        temps_pour_pi = sum(self.temps_pi)/len(self.temps_pi) # Calcul du temps pour pi
        temps_pour_M = sum(self.temps_M)/len(self.temps_M) # Calcul du temps pour M
        temps_pour_ergodic = sum(self.temps_ergodic)/len(self.temps_ergodic) # Calcul du temps pour ergodic
        temps_pour_point_fixe = sum(self.temps_point_fixe)/len(self.temps_point_fixe) # Calcul du temps pour point_fixe
        print("Temps pour Pi : ", temps_pour_pi, "\nTemps pour M : ", temps_pour_M, "\nTemps pour ergodic : ", temps_pour_ergodic, "\nTemps pour point fixe : ", temps_pour_point_fixe)
        return temps_pour_pi, temps_pour_M, temps_pour_ergodic, temps_pour_point_fixe
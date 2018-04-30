import numpy as np


class CdMConvergence():

    def __init__(self, cdm):
        self.cdm = cdm

    def convergence_pi_n(self, epsilon):
        """
        Méthode qui va calculer la convergence de pi n et s'arrête lorsque la
        différence entre n et n-1 est assez faible
        :return:
        """
        if not self.cdm.is_ergodic():
            return False
        array = np.zeros((1, len(self.cdm.get_states())))
        array_n_minus_one = array.copy()
        # print(array)
        # On initialise pi(0)
        for key, value in self.cdm.get_initial_distribution().items():
            array[0][int(key)-1]=value
        for i in range(100): # Itération (la valeur peut être modifié)
            # Multiplication de la matrice array avec la matrice de transition
            array = np.dot(array, self.cdm.get_transition_matrix())
            # print(array)
            # On verifie si la différence entre les deux matrices est assez faible (selon epsilon)
            if self.cdm.check_array_equals(array[0], array_n_minus_one[0], epsilon):
                # print(i, array[0])
                return True, i, array
            array_n_minus_one=array.copy()
        return False, i, array[0]

    def convergence_M_n(self, epsilon):
        """
        Méthode qui calcul la convergence de Mn
        :return: Booleen si on a réussi à converger, la position à laquelle
        on a convergé, le point
        """
        if not self.cdm.is_ergodic():
            return False
        array = self.cdm.get_transition_matrix()
        array_n_minus_one = array.copy()
        # print(array.shape)
        for i in range(100):
            array=np.dot(array, array)
            # print(array)
            if self.check_matrix_converge(array, array_n_minus_one, epsilon):
                # print(i, array)
                return True, i, array
            array_n_minus_one=array.copy()
        return False, i, array[0]

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
        pi_n = self.convergence_pi_n(0.000001)
        M_n = self.convergence_M_n(0.000001)
        print(pi_n[2])
        print("Vecteur propre de M pour la valeur propre 1 : ", np.dot(pi_n[2][0], M_n[2]))

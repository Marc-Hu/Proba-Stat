import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import sys, os

# Main definition - constants
menu_actions  = {}  
 
# =======================
#     MENUS FUNCTIONS
# =======================
 
# Main menu
def main_menu():
    os.system('clear')
    
    print ("Projet 3I005,\n")
    print ("Statistique en Bioinformatique,\n")
    print ("Analyse statique d'une famille de protéines,\n")
    print ("Par Chaouche Sabrina & Hu Marc\n")
    print ("Choisissez la partie du projet que vous souhaitez:")
    print ("1. Première partie") #Va appeler le premier menu
    print ("2. Deuxième partie") # Va appeler le deuxième menu
    print ("\n0. Quitter") #Quitte le programme
    choice = input(" >>  ") #Saisie clavier de l'utilisateur
    exec_menu(choice)
 
    return
 
# Execute menu
def exec_menu(choice):
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()#Exécution de la fonction du choix
        except KeyError:
            print ("Sélection invalide, veuillez réessayer\n")
            menu_actions['main_menu']()
    return
 
# Menu 1
def menu1():
    print ("Première partie\n")
    print ("1. Fonction 1 (ni(a), wi(a))")
    print ("2. Fonction 2 (Entropie)")
    print ("3. Fonction 3 (Paramètre f(o)(a))")
    print ("4. Fonction 4 (Log-vraisemblance)\n")
    print ("9. Retour")
    print ("0. Quitter")
    choice = input(" >>  ")
    if choice=="1" or choice=="2" or choice=="3" or choice=="4":
    	menu1_sub(choice) #Appel à la fonction si c'est entre 1 et 4
    else: 
    	exec_menu('9') #Sinon revient sur le menu principal
    return

#Choix du sous-menu 1
def menu1_sub(choice):
	print("En attente de calcul pour la matrice d'entraînement et les wia...")
	train = read_file("Dtrain.txt")
	matrix_train = matrix_bio(train)
	matrix_wia=matrix_wi_a(matrix_train)
	if choice=="1":
		position="-1"
		while int(position)<0 or int(position)>47:
			print ("Quelle position?")
			position=input(" >>   ")
		acide = "0"
		while acide not in ARRAY_ACIDE :
			print("Quel acide aminé?")
			acide=input(" >>   ")
		fonction_1(matrix_train, int(position), str(acide))
	elif choice=='2':
		position="-1"
		while not position.isdigit() or int(position)<0 or int(position)>47:
			print ("Quelle position?")
			position=input(" >>   ")
		position=int(position)
		print("Si à la position " ,position, " est de " ,fonction_2(matrix_wia)[0][position])
	elif choice=='3':
		acide = "0"
		while acide not in ARRAY_ACIDE :
			print("Quel acide aminé?")
			acide=input(" >>   ")
		position=ARRAY_ACIDE.index(acide)
		print("f(0) de", acide , "est",fonction_3(matrix_wia, matrix_train[0]))
	else :
		matrix_test = getMatrix_test()
		fonction_4(matrix_test, matrix_wia)
	print("\n\nAppuyer sur une touche pour revenir au menu\n")
	input()
	exec_menu('9')
	return
 
# Menu 2
def menu2():
    print ("Première partie\n")
    print ("1. Fonction 1 (wi(a))")
    print ("2. Fonction 2 (nij(a, b) et wij(a, b))")
    print ("3. Fonction 3 (Information mutuelle Mij)")
    print ("4. Fonction 4 (Courbe fraction selon le nombre de paires)\n")
    print ("9. Retour")
    print ("0. Quitter")
    choice = input(" >>  ")
    if choice=="1" or choice=="2" or choice=="3" or choice=="4":
    	menu2_sub(choice)
    else:
    	exec_menu('9')
    return

def menu2_sub(choice):
	print("En attente de calcul pour la matrice d'entraînement et wia...")
	train = read_file("Dtrain.txt")
	matrix_train = matrix_bio(train)
	matrix_wia=matrix_wi_a(matrix_train)
	if choice=="1":
		position="-1"
		while int(position)<0 or int(position)>47:
			print ("Quelle position?")
			position=input(" >>   ")
		acide = "0"
		while acide not in ARRAY_ACIDE :
			print("Quel acide aminé?")
			acide=input(" >>   ")
		function_1(matrix_train, int(position), str(acide))
	elif choice=='2':
		positioni="-1"
		while int(positioni)<0 or int(positioni)>47:
			print ("Quelle position i?")
			positioni=input(" >>   ")
		positionj="-1"
		while int(positionj)<0 or int(positionj)>47:
			print ("Quelle position j (strictemennt supérieur à i)?")
			positionj=input(" >>   ")
		acidea = "0"
		while acidea not in ARRAY_ACIDE :
			print("Quel acide aminé a?")
			acidea=input(" >>   ")
		acideb = "0"
		while acideb not in ARRAY_ACIDE :
			print("Quel acide aminé b?")
			acideb=input(" >>   ")
		print("En cours de calcul... (Environ 2min selon l'ordinateur)")
		acidea=ARRAY_ACIDE.index(acidea)
		acideb=ARRAY_ACIDE.index(acideb)
		res=function_2_bis(matrix_train, int(positioni), int(positionj), acidea, acideb) #renvoie un tuple
		print("nij =", res[0], "wij =", res[1])
	elif choice=='3':
		matrice_paire_acide, matrice_paire_position=matrice_paire_acide_postion()
		positioni="-1"
		while int(positioni)<0 or int(positioni)>47:
			print ("\nQuelle position i?")
			positioni=input(" >>   ")
		positionj="-1"
		while int(positionj)<0 or int(positionj)>47:
			print ("Quelle position j (strictemennt supérieur à i)?")
			positionj=input(" >>   ")
		print("Calcul de Wij...")
		wij=function_2(matrix_train)[1]
		print("\nCalcul de l'information mutuelle Mij...\n")
		positioni=int(positioni)
		positionj=int(positionj)
		res=function_3_bis(wij, matrix_wia, matrice_paire_acide, matrice_paire_position, positioni, positionj)
		print("L'information mutuelle pour i =", positioni, "est j =", positionj," est", res)
	else:
		print("Calcul de Wij...")
		wij=function_2(matrix_train)[1]
		matrice_paire_acide, matrice_paire_position=matrice_paire_acide_postion()
		print("\nCalcul de l'information mutuelle Mij...\n")
		mij=function_3(wij, matrix_wia, matrice_paire_acide, matrice_paire_position)
		function_4(mij, matrice_paire_position)
	print("\n\nAppuyer sur une touche pour revenir au menu\n")
	input()
	exec_menu('9')
	return

# Back to main menu
def back():
    menu_actions['main_menu']()
 
# Exit program
def exit():
    sys.exit()
 
# =======================
#    MENUS DEFINITIONS
# =======================
 
# Menu definition
menu_actions = {
    'main_menu': main_menu,
    '1': menu1,
    '2': menu2,
    '9': back,
    '0': exit,
}
 
# Liste contenant les acides; à utiliser tout au long du projet.
ARRAY_ACIDE = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
M = 5643
L = 48

##
#	I. Données
##

##
# Fonctions pour la lecture des données de la première partie et leurs organisation dans une matrice.
##

# Fonction pour récupérer les données d'un fichier
def read_file(fname):
	res = []
	f = open(fname,'rb')
	raw_file = f.readlines()
	f.close()
	for i in range(len(raw_file)):
		raw_file[i] = raw_file[i].decode('utf-8')
	for i  in range(int(len(raw_file)/2)): # Ne récupérer que les lignes impaires représentant les protéines en comptant depuis 0.
		res.append(raw_file[i*2+1][0:len(raw_file[i*2+1])-1]) #!!!
	#print(res)
	return res

#Fonction pour initialiser la matrice training
def matrix_bio(train):
	char_array = np.chararray((len(train), L))
	char_array[:] = '*'
	for i in range(len(train)):
		for j in range(L):
			char_array[i][j] = train[i][j]
	#print(char_array)
	return char_array

##
#	II. Modélisation par PSWM
##

##
# Première fonction: Pour chaque position (colonne) i = 0, ..., L−1 et chaque acide aminée a ∈ A(le trou compris), on calcule le nombre d’occurence ni(a) (équation 1) et le poid ωi(a) (équation 3).
##



# Fonction pour calculer le nombre d'occurences d'un acide anminé donné sur une position donnée.
def ni_a(matrix_train, i, a):
	result = 0
	column_position = matrix_train[:,i]
	for j in range(len(column_position)):
		if column_position[j].decode('utf-8') == a :
			result += 1
	return result

# Fonction pour calculer le nombre d'occurences de chaque acide anminé donné sur chaque position donnée.
def matrix_ni_a(matrix_train):
	result = np.zeros((L, 21))
	for i in range(matrix_train.shape[1]):
		for j in range(len(ARRAY_ACIDE)):
			result[i][j] = ni_a(matrix_train, i, ARRAY_ACIDE[j])
	return result
	
# Fonction pour calculer le poids d'un acide aminé donné sur une position donnée.
"""def wi_a(matrix_ni_a, i, a):
	return (matrix_ni_a[i][ARRAY_ACIDE.index(a)] + 1)/(M + 21)
"""
def wi_a(matrix_train, i, a):
	return (ni_a(matrix_train, i, a) + 1)/(M + 21)

# Fonction pour calculer le poids pour chaque acide aminé sur une position donnée i.
def matrix_wi_a(matrix_train):
	result = np.zeros((L, 21)) # Initialiser une matrice à 0 de meme dimensions que ni_a
	for i in range (L):
		for j in range(21):
			result[i][j] = wi_a(matrix_train, i, ARRAY_ACIDE[j])
	return result
	
def fonction_1(matrix_train, i, a):
	nia = ni_a(matrix_train, i, a)
	wia = wi_a(matrix_train, i, a)
	print("Nombre d'occurences de l'acide aminé", a, "à la position", i, "est:", nia)
	print("Le poids de l'acide aminé", a, "à la position", i, "est:", wia)
	
	
##
#  Deuxième fonction: Pour chaque position i = 0, ..., L−1, déterminer l’entropie rélative Si (équation(4)), et pour les trois positions plus conservées aussi les acides aminées conservées (équation (5)). Tracer l’entropie rélative en fonction de la position i.
##

# Fonction pour calculer l'entropie rélative pour chaque position.
def si(wi_a):
	result = np.zeros((1, wi_a.shape[0])) # Initialiser la matrice qui doit contenir le résultat 
	for i in range(wi_a.shape[0]): # Pour chaque position dans les 48 existantes
		for j in range(wi_a.shape[1]): # Pour chaque acide aminée
			result[0][i] = result[0][i] + wi_a[i][j] * np.log2(wi_a[i][j])
		result[0][i] = result[0][i] + np.log2(21)
	return result	

# Fonction pour renvoyer les 3 acides les plus conservés.
def trois_acides_plus_conserves(wi_a, trois_position_conserve):
	result = []
	for i in range(3):
		result.append(ARRAY_ACIDE[np.argmax(wi_a[int(trois_position_conserve[0][i])])])
	print("Acides les plus conservés aux positions les plus conservés : ", result)
	return result

# Fonction pour afficher l'entropie
def affiche_entropie(si_a):
	x = np.arange(L);
	plt.title("Entropie rélative en fonction de la position i")
	plt.xlabel("Positions")
	plt.ylabel("Entropie rélative S")
	plt.plot(si_a[0])
	plt.show()
	
def fonction_2(wi_a):
	si_a = si(wi_a) # Matrice d'entropie rélative
	trois_position_conserve = np.zeros((1, 3))# Initialiser la matrice qui va contenir les 3 positions les plus conservées.
	position = np.argsort(si_a[0])[::-1] # Contient les indices des valeurs d'entropie triées.
	for i in range(trois_position_conserve.shape[1]):
		trois_position_conserve[0][i] = position[i]
	print("Les trois positions les plus conservés : ", trois_position_conserve[0])
	trois_acides_plus_conserves(wi_a, trois_position_conserve)
	affiche_entropie(si_a)
	return si_a

##
# Troisième fonction: Déterminer les paramètres f(0)(a) du modèle nul.
##

# Fonction pour calaculer un paramètre du modèle nul.
def param_modele_nul(wi_a, acide):
	result = 0.0
	index = ARRAY_ACIDE.index(acide)
	result = np.sum(wi_a[:,index])/L
	return result
	
def fonction_3(wi_a, b):
	result = 1
	for i in range(L):
		result *= param_modele_nul(wi_a, b[i].decode('utf-8'))
	return result

##
# Quatrième fonction: Déterminer ℓ(bi, ..., bi+L−1) (équation (9)) pour chaque sous-séquence de longueur L. Déterminer si il y a des sous-séquences de la famille definie par Dtrain. Tracer la log-vraisemblance en fonction de sa première position i = 0, ..., N − L.
##
	
# Récupération de la matrice test dans le fichier test_seq
def getMatrix_test():
	result = read_file("test_seq.txt")
	char_array = np.chararray((len(result[0])-L, L));
	char_array[:] = '*'
	for i in range(len(result[0])-L):
		for j in range(L):
			char_array[i][j] = result[0][j+i]
	return char_array
	
# Fonction pour tracer la log-vraisemblance des sous-séquences en fonction de sa première position i = 0, ..., N − L.
def affiche_log_vraisemblance(data):
	plt.title("Log-vraisemblance en fonction de la première position des sous-séquences du test")
	plt.xlabel("Position")
	plt.ylabel("log-vraisemblance")
	plt.plot(data)
	plt.show()

# Fonction pour calculer le log de vraisemblance 
def log_vraisemblance(wi_a, sequence):
	result = 0.0
	for i in range(L):
		acide = sequence[i].decode("UTF-8")
		result = result + np.log2(wi_a[i][ARRAY_ACIDE.index(acide)] / param_modele_nul(wi_a, acide))
	return result
	
# Fonction pour afficher le log-vraisemblance à une position
def log_vraisemblance_position(data, position):
	print("L(b",position,") = ", result[0][position])

	
def fonction_4(matrix_test, wi_a):
	result = np.zeros((1, matrix_test.shape[0]))
	for i in range(matrix_test.shape[0]):
		result[0][i] = log_vraisemblance(wi_a, matrix_test[i])
			
	for i in range(result.shape[1]):
		if result[0][i] > 0:
			print(matrix_test[i].decode('utf-8'), "\nest une sous-séquence de la famille définie par Dtrain")
			
	affiche_log_vraisemblance(result[0])
	return result
	

##
#	III. Co-écolution de résidues en contact
##

##
# Première fonction: Calculer wi(a).
##

def function_1(matrix_train, i, a):
	wia=wi_a(matrix_train, i, a)
	print("Le poids de l'acide aminé", a, "à la position", i, "est:", wia)

##
# Deuxième fonction: Pour chaque paire de positions 1≤i<j≤L et chaque combinaison d’acides aminées a, b ∈A(le trou compris), calculer le nombre d’occurence nij(a, b) (équation (10)) et le poids ωij(a, b) (équation(11)).
##

# Fonction pour construire deux patrices: l'une contient les paires d'acides (a,b), l'autre les paires de positions possibles (i,j).
def matrice_paire_acide_postion():
	matrice_paire_acide = []
	for i in range(len(ARRAY_ACIDE)): # On construit le tableau des paires d'acide
		for j in range(len(ARRAY_ACIDE)):
			res = ARRAY_ACIDE[i] + ARRAY_ACIDE[j]
			matrice_paire_acide.append(res)
			
	matrice_paire_position = []
	for i in range(L-1):	# On construit le tableau des paires de position
		for j in range(i+1, L):
			matrice_paire_position.append((i, j))
	
	return matrice_paire_acide, matrice_paire_position

# Fonction pour calculer les nij.
def nij(matrix_train, matrice_paire_acide, matrice_paire_position):
	result = np.zeros((len(matrice_paire_acide), len(matrice_paire_position)))
	for i in range(result.shape[1]):
		for j in range(matrix_train.shape[0]):
			acid_a = matrix_train[j][matrice_paire_position[i][0]].decode("utf-8")
			acid_b = matrix_train[j][matrice_paire_position[i][1]].decode("utf-8")
			index = matrice_paire_acide.index(acid_a + acid_b)
			result[index][i] = result[index][i]+1
	return result


# Fonction pour calculer le poids wij
def wij(nij):
	result = np.zeros(nij.shape)
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			# Application de la formule 11
			result[i][j] = (nij[i][j]+(1/21))/(M+21)
	return result	
	
def function_2(matrix_train):
	matrice_paire_acide, matrice_paire_position = matrice_paire_acide_postion()
	n_ij = nij(matrix_train, matrice_paire_acide, matrice_paire_position)
	w_ij = wij(n_ij)
	return n_ij, w_ij 	

# Fonction pour retourner le nombres d'occurnces et le poids à 2 positions i et j spécifiques pour 2 acides aminés bien définis.
def function_2_bis(matrix_train, i, j, a, b):
	matrice_paire_acide, matrice_paire_position = matrice_paire_acide_postion()
	nij_result = nij(matrix_train, matrice_paire_acide, matrice_paire_position)
	wij_result = wij(nij_result)
	i = matrice_paire_acide.index(ARRAY_ACIDE[int(a)] + ARRAY_ACIDE[int(b)])
	j = matrice_paire_position.index((i, j))
	return (nij_result[i][j], wij_result[i][j])

##
# Troisième fonction: Pour chaque paire de positions 0≤i<j≤L−1, calculer l’information mutuelle Mij (équation (12)).
##	
def function_3(wij, wi_a, matrice_paire_acide, matrice_paire_position):
	result = np.zeros((1, len(matrice_paire_position)))
	for i in range(len(matrice_paire_position)): # On parcours toutes les colonnes
		pos_i = matrice_paire_position[i][0] # On récupère la position i
		pos_j = matrice_paire_position[i][1] # On récupère la position j
		for j in range(wij.shape[0]):	# On parcours toutes les lignes
			pos_a = ARRAY_ACIDE.index(matrice_paire_acide[j][0]) # On récupère la position de l'acide a (index)
			pos_b = ARRAY_ACIDE.index(matrice_paire_acide[j][1]) # On récupère la position de l'acide b (index)
			# Application de la formule 12.
			result[0][i] = result[0][i]+(wij[j][i]*np.log2(wij[j][i]/(wi_a[pos_i][pos_a]*wi_a[pos_j][pos_b])))
	return result

# Fonction pour calculer l’information mutuelle Mij pour i et j donnés.
def function_3_bis(wij, wi_a, matrice_paire_acide, matrice_paire_position, index_i, index_j):
	result = np.zeros((1, len(matrice_paire_position)))
	for i in range(len(matrice_paire_position)): # On parcours toutes les colonnes
		pos_i = matrice_paire_position[i][0] # On récupère la position i
		pos_j = matrice_paire_position[i][1] # On récupère la position j
		
		for j in range(wij.shape[0]):	# On parcours toutes les lignes
			pos_a = ARRAY_ACIDE.index(matrice_paire_acide[j][0]) # On récupère la position de l'acide a (index)
			pos_b = ARRAY_ACIDE.index(matrice_paire_acide[j][1]) # On récupère la position de l'acide b (index)
			# Application de la formule 12.
			result[0][i] = result[0][i] + (wij[j][i] * np.log2(wij[j][i] / (wi_a[pos_i][pos_a]*wi_a[pos_j][pos_b])))
	return result[0][matrice_paire_position.index((index_i, index_j))]
	
##
# Quatrième fonction
#

# Fonction pour lire le fichier des distances.
def read_file_distance():
	res = []
	f = open("distances.txt",'rb')
	raw_file = f.readlines()
	f.close()
	for i in range(len(raw_file)):
		res.append(raw_file[i].decode("utf-8").split())
	return res

# Fonction pour afficher les fractions.
def affiche_fraction(data, axis):
	x = np.arange(10);
	for i in range(len(axis)):
		axis[i]=int(axis[i])
	plt.title("Fraction des paires sélectionnées qui ont une distance < 8 par rapport au nombre de paires")
	plt.xlabel("Nombre de paires considérées")
	plt.ylabel("Fraction des paires")
	plt.xticks(x, axis)
	plt.plot(data)
	plt.show()

#
def fraction(mij_50_grand, distances, matrice_paire_position):
	interval = [0] * 10 # Choisir des intervalles avec un pas de 10 -> 5 intervalles
	result = [0.0] * 10 # Contient la fraction qui correspond à chaque intervalle parmi les 5 choisis.
	
	for i in range(len(interval)): # Rennomer les intervalles avec les bons indices pour l'affichage.
		interval[i] = (len(mij_50_grand)/10) + (len(mij_50_grand)/10)*i

	for i in range(len(interval)):
		res = mij_50_grand[0:int(interval[i])] # Récupérer les valeurs mij correspond à l'intervalle "i".
		for j in range(len(res)):
			index = matrice_paire_position.index((int(res[j][0]), int(res[j][1])))
			if float(distances[index][2]) < 8.0:
				result[i] = result[i] + 1.0
		result[i] = result[i] / interval[i] # Calculer la fraction correspondant à l'intervalle "i".
	return result, interval

	
def function_4(mij, matrice_paire_position):
	index = np.argsort(mij) # Contient les indices des valeurs Mij triées dans un ordre croissant.
	result = []
	for i in range(50): # Récupérer les 50 plus grandes valeurs
		result.append(matrice_paire_position[index[0][len(mij[0])-1-i]])
	distances = read_file_distance()
	result_fraction, intervalles = fraction(result, distances, matrice_paire_position)
	affiche_fraction(result_fraction, intervalles)

	
if __name__ == '__main__':
	# Launch main menu
	main_menu()
	# train = read_file("Dtrain.txt")
	# matrix_train = matrix_bio(train)
	# print(ni_a(matrix_train, 0, '-'))
	# matrix_nia = matrix_ni_a(matrix_train)
	# print(wi_a(matrix_train, 0,'-'))
	# matrix_wia = matrix_wi_a(matrix_train)
	# print(matrix_wia[0][20])
	# # print(fonction_3(matrix_wia, matrix_train[0]))
	# matrix_test = getMatrix_test()
	# # fonction_4(matrix_test, matrix_wia)
	# #fonction_1(matrix_train, 0, '-')
	# # print(matrice_paire_acide_postion())
	# #function_2(matrix_train)
	

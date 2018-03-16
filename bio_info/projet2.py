import numpy as np
import copy as cp
import matplotlib.pyplot as plt

ARRAY_ACIDE=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
#Fonction qui va récupérer les données d'un fichier
def read_file(fname):
	res=[]
	f = open(fname,'rb')
	raw_file = f.readlines()
	f.close()
	for i in range(len(raw_file)):
		raw_file[i]=raw_file[i].decode('utf-8')
	for i  in range(int(len(raw_file)/2)):
		res.append(raw_file[i*2+1][0:len(raw_file[i*2+1])-1])
	# print(res)
	return res

#Fonction qui va initialiser la matrice training
def matrix_bio(train):
	char_array=np.chararray((len(train), 48));
	char_array[:] = '*'
	for i in range(len(train)):
		for j in range(len(train[i])):
			char_array[i][j]=train[i][j]
	return char_array;

##
#	I. Données
##

##
#	Fonction 1
##

# Fonction qui va récupérer ni_a et wi_a
def fonction_1(matrix_train):
	M=matrix_train.shape[0]
	res_ni_a=ni_a(matrix_train)
	res_wi_a=wi_a(res_ni_a,M)
	return res_ni_a, res_wi_a

#Fonction ni_a qui va initialiser la matrice ni_a
def ni_a(matrix_train):
	j=0;
	test=0
	#Variable qui permet de trouver facilement l'indice à laquel se trouve un acide aminé
	acide=np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-'])
	result=np.zeros((48, len(acide))) #Initialise une matrice de 48 par 21
	
	for i in range(matrix_train.shape[1]):
		unique, counts = np.unique(matrix_train[:,i], return_counts=True) # Compte le nombre d'acide i sur une colonne
		res=dict(zip(unique, counts)) #On a un dictionnaire key, value où key est l'acide et value le nombre d'acide key dans la colonne
		# print(res);
		for key, value in res.items(): #Pour tous les acides trouvés
			# print(key)
			j=ARRAY_ACIDE.index(key.decode('utf-8'));#On récupère l'indice ou ce trouve la clé
			result[i][j]=value#Et on ajoute sa valeur dans le resultat
			test=test+value
	# print(test/48)
	return result;

# Fonction qui va calculer wi_a
def wi_a(matrix_ni_a, M):
	result_shape=matrix_ni_a.shape;#Récupère les dimensions de la matrice ni_a 
	result=np.zeros(result_shape)#Initialise une matrice à 0 de dimension ni_a
	# print(result.shape[0])
	for i in range (result_shape[0]):
		for j in range(result_shape[1]):
			result[i][j]=(matrix_ni_a[i][j]+1)/(M+result_shape[1]) #Formule #3
	# print("Wo(-) : " ,result[0][20]);
	return result;

##
#	Fonction 2
##

#Fonction qui va récupérer si_a et les trois positions les plus conservé
def fonction_2(wi_a):
	si_a=si(wi_a)
	copy_si=cp.copy(si_a) #On fait une copie de si car on va prendre les valeurs max donc à chaque val max on va mettre à 0 la précédente valeur max
	# print(np.sort(si_a))
	trois_position_conserve=np.zeros((1, 3))#On initialise notre matrice qui va contenir les 3 acides les plus grands
	# print(trois_position_conserve.shape, copy_si)
	position=np.argsort(copy_si[0])[::-1]
	# print(position)
	for i in range(trois_position_conserve.shape[1]):
		trois_position_conserve[0][i]=position[i]
	print("Les trois positions les plus conservés : ", trois_position_conserve[0])
	affiche_entropie(si_a)
	trois_acide_plus_conserve(wi_a, trois_position_conserve)
	return trois_position_conserve

#Fonction qui calcul wi_a
def si(wi_a):
	# print(wi_a.shape[1])
	q=wi_a.shape[0]
	result=np.zeros((1, q))# On initialise la matrice de taille 1 par le nb d'acide
	vecteur = np.zeros((1, 48))
	# print(wi_a)
	for i in range(q):#Pour chaque acide 
		for j in range(21):
			vecteur[0][i] = vecteur[0][i] + wi_a[i][j] * np.log2(wi_a[i][j])
		vecteur[0][i] = vecteur[0][i] + np.log2(21)
		#res=np.sum(wi_a[i,:])#On va faire la somme de la colonne i
		# print(res)
		#result[0][i]=np.log2(q)+res*np.log2(res)# Formule #4
	# print(vecteur)
	return vecteur

def affiche_entropie(si_a):
	x=np.arange(48);
	plt.plot(si_a[0])
	plt.show()

#Fonction qui renvoit les 3 acides les plus conservé
def trois_acide_plus_conserve(wi_a, trois_position_conserve):
	result=[]
	for i in range(3):
		# print(np.argmax(wi_a[int(trois_position_conserve[0][i])]))
		result.append(ARRAY_ACIDE[np.argmax(wi_a[int(trois_position_conserve[0][i])])])
	print("Acides les plus conservés aux positions les plus conservés : ", result)
	return result

##
#	Function 3
##

#Renvoi le f(o) d'un acide donné
def function3(acide, wi_a):
	result=0
	index = ARRAY_ACIDE.index(acide)
	result= np.sum(wi_a[:,index])/48
	return result

##
#	Fonction 4
##	

def getMatrix_test():
	res=read_file("test_seq.txt")
	print(res[0])
	char_array=np.chararray((len(res[0])-48, 48));
	char_array[:] = '*'
	for i in range(len(res[0])-48):
		for j in range(48):
			char_array[i][j]=res[0][j+i]
	print(char_array)
	return char_array

def function4(matrix_test, wi_a):
	print(np.shape(matrix_test))
	result=[0.0]*np.shape(matrix_test)[0]
	for i in range(np.shape(matrix_test)[0]):
		for j in range(48):
			acide=matrix_test[i][j].decode("UTF-8")
			result[i]=result[i]+np.log2(wi_a[j][ARRAY_ACIDE.index(acide)]/function3(acide, wi_a))
	print(result)
	plt.plot(result)
	plt.show()

##
#	II.Co-écolution de résidues en contact
##

def fonct_1(matrix_train):
	return fonction_1(matrix_train)[1]

#Fonction qui calcule le nombre d'occurence nij
def nij(matrix_train, matrice_paire_acide, matrice_paire_position):
	result=np.zeros((len(matrice_paire_acide), len(matrice_paire_position)))
	# print(result.shape)
	for i in range(result.shape[1]):
		for j in range(matrix_train.shape[0]):
			# print(matrix_train[j][matrice_paire_position[i][0]])
			acid_a=matrix_train[j][matrice_paire_position[i][0]].decode("utf-8")
			acid_b=matrix_train[j][matrice_paire_position[i][1]].decode("utf-8")
			if ARRAY_ACIDE.index(acid_a)>ARRAY_ACIDE.index(acid_b):
				sub=acid_a
				acid_a=acid_b
				acid_b=sub
			index=matrice_paire_acide.index(acid_a+acid_b)
			# print(acid_a, acid_b, index, i)
			result[index][i]=result[index][i]+1
	print(result)
	return result

#Fonction qui calcul le poids wij
def wij(nij, M):
	result=np.zeros(nij.shape)
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			#Application de la formule 11
			result[i][j]=(nij[i][j]+(1/nij.shape[1]))/(M+nij.shape[1])
	# print(result)
	return result

#Fonction qui calcule le nombre d'occurence nij et le poids nij
def fonct_2(matrix_train):
	matrice_paire_acide=[]
	for i in range(len(ARRAY_ACIDE)): #On construit le tableau des paires d'acide
		for j in range(i, len(ARRAY_ACIDE)):
			res=ARRAY_ACIDE[i]+ARRAY_ACIDE[j]
			matrice_paire_acide.append(res)
	matrice_paire_position=[]
	for i in range(47):	#On construit le tableau des paires de position
		for j in range(i+1, 48):
			matrice_paire_position.append((i, j))
	nij_result=nij(matrix_train, matrice_paire_acide, matrice_paire_position)
	print(matrix_train.shape)
	wij_result=wij(nij_result, matrix_train.shape[0])
	return nij_result, wij_result, matrice_paire_acide, matrice_paire_position

def fonct_3(wij, wi_a, matrice_paire_acide, matrice_paire_position):
	result=np.zeros((1, len(matrice_paire_position)))
	print("test", wi_a.shape)
	for i in range(len(matrice_paire_position)): #On parcours toutes les colonnes
		pos_i=matrice_paire_position[i][0] #On récupère la position i
		pos_j=matrice_paire_position[i][1] #On récupère la position j
		# print(pos_a,'+', pos_b)
		for j in range(wij.shape[0]):	#On parcours toutes les lignes
			pos_a=ARRAY_ACIDE.index(matrice_paire_acide[j][0]) #On récupère la position de l'acide a (index)
			pos_b=ARRAY_ACIDE.index(matrice_paire_acide[j][1]) #On récupère la position de l'acide b (index)
			# print(pos_a,' ',pos_b)
			result[0][i]=result[0][i]+(wij[j][i]*np.log2(wij[j][i]/(wi_a[pos_i][pos_a]*wi_a[pos_j][pos_b])))
	print(result[0][0])
	return result

if __name__ == '__main__':
	train=read_file("Dtrain.txt")
	matrix_train=matrix_bio(train)
	# res_fonction_1=fonction_1(matrix_train)
	# # res_fonction_2=fonction_2(res_fonction_1[1])
	# matrix_test=getMatrix_test()
	# function4(matrix_test, res_fonction_1[1])
	wi_a=fonct_1(matrix_train)
	nij, wij, matrice_paire_acide, matrice_paire_position=fonct_2(matrix_train)
	fonct_3(wij, wi_a, matrice_paire_acide, matrice_paire_position)
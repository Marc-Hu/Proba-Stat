import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import sys

# Liste contenant les acides; à utiliser tout au long du projet.
ARRAY_ACIDE = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

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
	# print(res)
	return res

#Fonction pour initialiser la matrice training
def matrix_bio(train):
	char_array = np.chararray((len(train), 48))
	char_array[:] = '*'
	for i in range(len(train)):
		for j in range(48):
			char_array[i][j] = train[i][j]
	return char_array

##
#	I. Données
##

##
# Première fonction: Pour chaque position (colonne) i = 0, ..., L−1 et chaque acide aminée a ∈ A(le trou compris), on calcule le nombre d’occurence ni(a) (équation 1) et le poid ωi(a) (équation 3).
##

# Fonction qui va récupérer ni_a et wi_a
def fonction_1(matrix_train, position, acide):
	M=matrix_train.shape[0]
	res_ni_a=ni_a(matrix_train)
	res_wi_a=wi_a(res_ni_a,M)
	# print(res_ni_a[position][acide], res_wi_a[acide][acide])
	return res_ni_a[position][acide], res_wi_a[acide][acide]

def fonction_1_bis(matrix_train):
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
	trois_acide_plus_conserve(wi_a, trois_position_conserve)
	affiche_entropie(si_a)
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
	plt.title("Entropie relative en fonction de la position i")
	plt.xlabel("Positions")
	plt.ylabel("Entropie relative S")
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
#Récupération de la matrice test dans le fichier test_seq
def getMatrix_test():
	res=read_file("test_seq.txt")
	# print(res[0])
	char_array=np.chararray((len(res[0])-48, 48));
	char_array[:] = '*'
	for i in range(len(res[0])-48):
		for j in range(48):
			char_array[i][j]=res[0][j+i]
	# print(char_array)
	return char_array

#Equation 9 et affichage dans un plot
def function4(matrix_test, wi_a, position):
	# print(np.shape(matrix_test))
	result=[0.0]*np.shape(matrix_test)[0]
	for i in range(np.shape(matrix_test)[0]):
		for j in range(48):
			acide=matrix_test[i][j].decode("UTF-8")
			result[i]=result[i]+np.log2(wi_a[j][ARRAY_ACIDE.index(acide)]/function3(acide, wi_a))
	print("L(b", position, ") = ", result[position])
	for i in range(len(result)):
		if result[i]>0:
			print(matrix_test[i], "est une sous-séquence de la famille définie par Dtrain")
	plt.title("Log-vraisemblance en fonction de la première position i=0, ...,N-L sur une séquence à tester (testseq.txt)")
	plt.xlabel("Positions i")
	plt.ylabel("l(bi)")
	plt.plot(result)
	plt.show()

##
#	II.Co-écolution de résidues en contact
##
#

def fonct_1_bis(matrix_train):
	return fonction_1_bis(matrix_train)[1]

def fonct_1(matrix_train, position, acide):
	return fonction_1(matrix_train, position, acide)[1]

#Fonction qui calcule le nombre d'occurence nij
def nij(matrix_train, matrice_paire_acide, matrice_paire_position):
	result=np.zeros((len(matrice_paire_acide), len(matrice_paire_position)))
	# print(result.shape)
	for i in range(result.shape[1]):
		for j in range(matrix_train.shape[0]):
			# print(matrix_train[j][matrice_paire_position[i][0]])
			acid_a=matrix_train[j][matrice_paire_position[i][0]].decode("utf-8")
			acid_b=matrix_train[j][matrice_paire_position[i][1]].decode("utf-8")
			# if ARRAY_ACIDE.index(acid_a)>ARRAY_ACIDE.index(acid_b):
			# 	sub=acid_a
			# 	acid_a=acid_b
			# 	acid_b=sub
			index=matrice_paire_acide.index(acid_a+acid_b)
			# print(acid_a, acid_b, index, i)
			result[index][i]=result[index][i]+1
	# print(result)
	return result

#Fonction qui calcul le poids wij
def wij(nij, M):
	result=np.zeros(nij.shape)
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			#Application de la formule 11
			result[i][j]=(nij[i][j]+(1/21))/(M+21)
	# print(result)
	return result

def fonct_2_bis(matrix_train):
	matrice_paire_acide=[]
	for i in range(len(ARRAY_ACIDE)): #On construit le tableau des paires d'acide
		for j in range(len(ARRAY_ACIDE)):
			res=ARRAY_ACIDE[i]+ARRAY_ACIDE[j]
			matrice_paire_acide.append(res)
	matrice_paire_position=[]
	for i in range(47):	#On construit le tableau des paires de position
		for j in range(i+1, 48):
			matrice_paire_position.append((i, j))
	nij_result=nij(matrix_train, matrice_paire_acide, matrice_paire_position)
	# print(matrix_train.shape)
	wij_result=wij(nij_result, matrix_train.shape[0])
	return nij_result, wij_result, matrice_paire_acide, matrice_paire_position

#Fonction qui calcule le nombre d'occurence nij et le poids nij
def fonct_2(matrix_train, i, j, a, b):
	matrice_paire_acide=[]
	for i in range(len(ARRAY_ACIDE)): #On construit le tableau des paires d'acide
		for j in range(len(ARRAY_ACIDE)):
			res=ARRAY_ACIDE[i]+ARRAY_ACIDE[j]
			matrice_paire_acide.append(res)
	matrice_paire_position=[]
	for i in range(47):	#On construit le tableau des paires de position
		for j in range(i+1, 48):
			matrice_paire_position.append((i, j))
	nij_result=nij(matrix_train, matrice_paire_acide, matrice_paire_position)
	# print(matrix_train.shape)
	wij_result=wij(nij_result, matrix_train.shape[0])
	i=matrice_paire_acide.index(ARRAY_ACIDE[int(a)]+ARRAY_ACIDE[int(b)])
	j=matrice_paire_position.index((i, j))
	return nij_result[i][j], wij_result[i][j]

def fonct_3_bis(wij, wi_a, matrice_paire_acide, matrice_paire_position):
	result=np.zeros((1, len(matrice_paire_position)))
	# print("test", wi_a.shape)
	for i in range(len(matrice_paire_position)): #On parcours toutes les colonnes
		pos_i=matrice_paire_position[i][0] #On récupère la position i
		pos_j=matrice_paire_position[i][1] #On récupère la position j
		# print(pos_a,'+', pos_b)
		for j in range(wij.shape[0]):	#On parcours toutes les lignes
			pos_a=ARRAY_ACIDE.index(matrice_paire_acide[j][0]) #On récupère la position de l'acide a (index)
			pos_b=ARRAY_ACIDE.index(matrice_paire_acide[j][1]) #On récupère la position de l'acide b (index)
			# print(pos_a,' ',pos_b)
			result[0][i]=result[0][i]+(wij[j][i]*np.log2(wij[j][i]/(wi_a[pos_i][pos_a]*wi_a[pos_j][pos_b])))
	return result

def fonct_3(w_ij, wia, matrice_paire_acide, matrice_paire_position, index_i, index_j):
	result=np.zeros((1, len(matrice_paire_position)))
	# print("test", wi_a.shape)
	for i in range(len(matrice_paire_position)): #On parcours toutes les colonnes
		pos_i=matrice_paire_position[i][0] #On récupère la position i
		pos_j=matrice_paire_position[i][1] #On récupère la position j
		# print(pos_a,'+', pos_b)
		for j in range(w_ij.shape[0]):	#On parcours toutes les lignes
			pos_a=ARRAY_ACIDE.index(matrice_paire_acide[j][0]) #On récupère la position de l'acide a (index)
			pos_b=ARRAY_ACIDE.index(matrice_paire_acide[j][1]) #On récupère la position de l'acide b (index)
			# print(pos_a,' ',pos_b)
			result[0][i]=result[0][i]+(w_ij[j][i]*np.log2(w_ij[j][i]/(wia[pos_i][pos_a]*wia[pos_j][pos_b])))
	return result[0][matrice_paire_position.index((index_i, index_j))]

def read_file_distance():
	res=[]
	f = open("distances.txt",'rb')
	raw_file = f.readlines()
	f.close()
	for i in range(len(raw_file)):
		# print(raw_file[i].decode("utf-8"))
		res.append(raw_file[i].decode("utf-8").split())
	# print(res)
	return res

def fonct_4(mij, matrice_paire_position):
	# print(len(mij[0]), len(matrice_paire_position))
	index=np.argsort(mij)
	result=[]
	# print(result[0][0], mij[0][index[0]])
	for i in range(50):
		result.append(matrice_paire_position[index[0][len(mij[0])-1-i]])
	distances=read_file_distance()
	fraction(result, distances)
	# print(result)

def fraction(m_grand, distances):
	interval=[0]*10
	result=[0.0]*10
	for i in range(len(interval)):
		interval[i]=(len(m_grand)/10)+(len(m_grand)/10)*i
	# print(interval, len(distances))
	for i in range(len(interval)):
		res=m_grand[0:int(interval[i])]
		for j in range(len(res)):
			index=matrice_paire_position.index((int(res[j][0]), int(res[j][1])))
			# print(distances[index])
			if float(distances[index][2])<8:
				result[i]=result[i]+1
		result[i]=result[i]/interval[i]
	# print(result)
	affiche_fraction(result, interval)

def affiche_fraction(data, axis):
	x=np.arange(10);
	for i in range(len(axis)):
		axis[i]=int(axis[i])
	plt.title("Fraction des paires sélectionnées qui ont une distances plus < 8 par rapport au nombre de paires")
	plt.xlabel("Nombre de paire considéré")
	plt.ylabel("Fraction des paires")
	plt.xticks(x, axis)
	plt.plot(data)
	plt.show()

def error():
	print("Je ne connais pas ces arguments. Veuillez vous référer au guide d'utilisation")

if __name__ == '__main__':
	train=read_file("Dtrain.txt")
	matrix_train=matrix_bio(train)
	if sys.argv[1]=='1': #Si le deuxième argument est 1 c'est à dire pour la première partie du projet
		res_fonction_1=fonction_1_bis(matrix_train)
		if sys.argv[2][0:9]=="fonction1": #SI le troisième argument demande la fonction 1
			couple=sys.argv[2][9:len(sys.argv[2])].split(",") #L'utilisateur devra ajouter des paramètre comme suit : fonction1"1,1"
			if len(couple)<2:#Si il n'y a pas les deux arguments avec le nom de la fonction alors il y a une erreur
				error()
			else :
				position=int(couple[0])
				acide=int(couple[1])
				res_fonction_1=fonction_1(matrix_train, position, acide)
				print("Nombre d'occurence à la position ", position ," pour l'acide aminée ", ARRAY_ACIDE[acide]," est de ",res_fonction_1[0],"\nEt son poids à cette position est de :", res_fonction_1[1])
		elif sys.argv[2][0:9]=="fonction2": #Fonction 2
			res_fonction_2=fonction_2(res_fonction_1[1])
		elif sys.argv[2][0:9]=="fonction3":#Fonction 3
			index=sys.argv[2][9:len(sys.argv[2])] #Récupère l'acide demandé
			if len(index)==0:
				error()
			else :
				result_function_3=function3(ARRAY_ACIDE[int(index)], res_fonction_1[1])
				print("Paramètre f(0) de l'acide ", ARRAY_ACIDE[int(index)]," : ",result_function_3)
		elif sys.argv[2][0:9]=="fonction4": #Si c'est pour la fonction4
			position=int(sys.argv[2][9:len(sys.argv[2])])#Récupère la sous-séquence demandé
			if position<1:
				error()
			else:
				matrix_test=getMatrix_test()
				function4(matrix_test, res_fonction_1[1], position)
		else:
			error()
	elif sys.argv[1]=='2': #Sinon si c'est pour la deuxième partie du projet
		if sys.argv[2][0:9]=="fonction1":
			couple=sys.argv[2][9:len(sys.argv[2])].split(",")#Récupère les deux arguments (position et acide)
			if len(couple)<2:
				error()
			else :
				position=int(couple[0])
				acide=int(couple[1])
				res_fonction_1=fonction_1(matrix_train, position, acide)
				wi_a=fonct_1(matrix_train, position, acide)
				print("Le poids à la position ", position ," pour l'acide aminée ", ARRAY_ACIDE[acide]," est de ",wi_a, '.')
		elif sys.argv[2][0:9]=="fonction2": 
			couple=sys.argv[2][9:len(sys.argv[2])].split(",") #Récupère 4 arguments (position i, position j, acide a et acide b)
			if len(couple)<4 or couple[0]>=couple[1]:
				error()
			else:
				res_fonction_2=fonct_2(matrix_train, couple[0], couple[1], couple[2], couple[3])
				print("Nombre d'occurence à le couple de position ", couple[0], " et ", couple[1] ," pour le couple d'acide aminée ", ARRAY_ACIDE[int(couple[2])], "est", ARRAY_ACIDE[int(couple[3])]," est de ",res_fonction_2[0],"\nEt son poids à cette position est de :", res_fonction_2[1])
			# nij, wij, matrice_paire_acide, matrice_paire_position=fonct_2(matrix_train)
		elif sys.argv[2][0:9]=="fonction3":
			couple=sys.argv[2][9:len(sys.argv[2])].split(",") #Récupère les deux arguments (position i et position j)
			if len(couple)<2:
				error()
			else:
				nij, wij, matrice_paire_acide, matrice_paire_position=fonct_2_bis(matrix_train)
				wi_a=fonct_1_bis(matrix_train)
				index_i=int(couple[0])
				index_j=int(couple[1])
				mij=fonct_3(wij, wi_a, matrice_paire_acide, matrice_paire_position, index_i, index_j)
				print("Information mutuelle M",index_i,index_j, " : ", mij)
		elif sys.argv[2][0:9]=="fonction4":
			wi_a=fonct_1_bis(matrix_train)
			nij, wij, matrice_paire_acide, matrice_paire_position=fonct_2_bis(matrix_train)
			mij=fonct_3_bis(wij, wi_a, matrice_paire_acide, matrice_paire_position)
			fonct_4(mij, matrice_paire_position)
		else :
			error()
	else :
		print("Il manque des arguments ou les arguments ne sont pas correct!")

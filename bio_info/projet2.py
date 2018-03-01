import numpy as np

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

#Fonction ni_a qui va initialiser la matrice ni_a
def ni_a(matrix_train):
	j=0;
	#Variable qui permet de trouver facilement l'indice à laquel se trouve un acide aminé
	array_acide=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
	acide=np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-'])
	result=np.zeros((48, len(acide)))
	for i in range(len(matrix_train[0])):
		unique, counts = np.unique(matrix_train[:,i], return_counts=True)
		res=dict(zip(unique, counts))
		for key, value in res.items():
			j=array_acide.index(key.decode('utf-8'));
			result[i][j]=value
	return result;

def wi_a(matrix_ni_a, M):
	result=np.zeros(matrix_ni_a.shape)
	# print(result.shape[0])
	result_shape=result.shape;
	for i in range (result_shape[0]):
		for j in range(result_shape[1]):
			result[i][j]=(matrix_ni_a[i][j]+1)/(M+result_shape[1])
	print(result);
	return result;

if __name__ == '__main__':
	train=read_file("Dtrain.txt")
	matrix_train=matrix_bio(train)
	M=matrix_train.shape[0]
	ni_a=ni_a(matrix_train)
	wi_a=wi_a(ni_a,M)
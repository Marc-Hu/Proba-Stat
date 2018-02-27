import numpy as np

#Calculer la scale_distance euclidienne entre l'indiv et le reste de la population
def scale_distance_Euclidienne(indiv , Data, sigma):
    return np.divide(np.sqrt(np.sum((indiv-Data)**2, axis=1)),2*sigma)

#Calculer la distance euclidienne entre l'indiv et le reste de la population
def distance_Euclidienne(indiv , Data):
    return np.sqrt(np.sum((indiv-Data)**2, axis=1))

#calculer la matrice corespendant à la formule exp -d(xi,xj)/2*sigma
def calcul_matrice_Distance_Exp(data, sigma):
    nbE = data.shape[0]
    #Calculer les distances entre les individus 
    MDistance = np.zeros(shape=(nbE,nbE))
    for i, indiv in zip(range(0,nbE-1,1),data): #le dernier individu on ne le prend pas en considération
        RowDi = scale_distance_Euclidienne(indiv, data[i+1:nbE,:], sigma)
        #print(RowDi)
        MDistance[i,(i+1):nbE] = RowDi
        MDistance[(i+1):nbE, i] = RowDi
    return np.exp((-1)*MDistance)

def calcul_Pij(MDistance):
    nbE = MDistance.shape[0]
    Pij = np.zeros(shape=(nbE,nbE))
    for i in range(0,nbE,1):
        for j in range(0,nbE,1):
            Pij[i,j] = MDistance[i,j] / (np.sum(MDistance[i,:])-MDistance[i,i])
    return Pij

def calcul_Qij(MDistanceY):
    nbE = MDistanceY.shape[0]
    Qij = np.zeros(shape=(nbE,nbE))
    for i in range(0,nbE,1):
        for j in range(0,nbE,1):
            Qij[i,j] = MDistanceY[i,j] / (np.sum(MDistanceY[i,:])-MDistanceY[i,i])
    return Qij


def calcul_GradiantY(Y, Pij, Qij, composante):
    Vy = np.zeros(shape=(Y.shape[0],2))
    for i in range(0, Y.shape[0]):
        somme=0
        somme2=0
        for j in range(0, Y.shape[0]):
            somme = somme + (Y[i,0] - Y[j,0])*(Pij[i,j]-Qij[i,j]+Pij[j,i]-Qij[j,i])
            somme2 = somme2 + (Y[i,1] - Y[j,1])*(Pij[i,j]-Qij[i,j]+Pij[j,i]-Qij[j,i])
        Vy[i,0]=2*somme
        Vy[i,1]=2*somme2
    return Vy

#size =(5....) 5 nbr de mail juste pour le test
data = np.random.choice([0, 1], size=(5,5), p=[1./2, 1./2]) #init random data
print(data)

MDistance = calcul_matrice_Distance_Exp(data, 0.5)
print('Matrice des distances')
print(MDistance)
Pij = calcul_Pij(MDistance)
print('Matrice des Pij')
print(Pij)
#genrer les yi

y = np.random.normal(0,0.5, size=(5,2))
print('Matrice des Y')
print(y)
#calculer les exp_distance entre les yi (il faut que le sigma =0.5 pour l'eliminer de l'équation car 2*0.5=1
MDistanceY = calcul_matrice_Distance_Exp(y, 0.5)
print('Matrice des distances Y')
print(MDistanceY)
Qij = calcul_Qij(MDistanceY)
print('Matrice des QIj')
print(Qij)

for i in range(0,100):
    y=calcul_GradiantY(y,Pij,Qij,0)
print('Y apres 100 intération')

#PS : pour le plot et l'affichage a la fin une fois que vous avez les Y labelisez les selon le label des données initiales
#genre par exemple si vous avez 100mail non spam il corresponde aux 100 premier ligne de Y et vice versa
print(y)

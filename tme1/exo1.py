import random;
import matplotlib.pyplot as plt

#Initialise un paquet de 52 carte constitue d'un nombre et d'une couleur
#Melange la liste
#Retourne la liste
def paquet():
	num = (1,2,3,4,5,6,7,8,9,10,11,12,13);
	couleur = ("C", "K", "P", "T");
	paquet = list();
	for i in range (len(num)):
		for j in range (len(couleur)):
			paquet.append((num[i], couleur[j]));
	random.shuffle(paquet);
	#print(paquet);
	return paquet;
	
#Compare 2 paquets de cartes et ajoute dans une liste les positions ou les cartes sont identiques
#Retourne la liste
def meme_position(p,q):
	meme_val_list = list();
	for i in range (52):
		if (p[i][0]==q[i][0] and p[i][1]==q[i][1]):
			meme_val_list.append(i);
	return meme_val_list;

def experience():
	nb_experience=10000;
	somme_precedent_experience=0;
	moyenne_position=[0.00] * nb_experience;
	resultat_position=[0.00] * nb_experience;
	paquet1=paquet();
	paquet2=paquet();
	meme_val_list=meme_position(paquet1, paquet2);
	moyenne_position[0]=len(meme_val_list);
	resultat_position[0]=moyenne_position[0];
	for i in range (1, nb_experience):
		paquet1=paquet();
		paquet2=paquet();
		meme_val_list=meme_position(paquet1, paquet2);
		moyenne_position[i]=float(len(meme_val_list));
		# print(len(meme_val_list));
		somme_precedent_experience=0;
		for j in range (i):
			somme_precedent_experience=somme_precedent_experience+moyenne_position[j];
		resultat_position[i]=float(somme_precedent_experience/i);
	return resultat_position;
			

def affichage_courbe(list):
	plt.plot(list);
	plt.xlabel('Evolution moyenne pour 10000');
	plt.show();
	
if __name__ == '__main__':
	#paquet1 = paquet();
	#paquet2 = paquet();
	#print(meme_position(paquet1, paquet2));
	proba_position=experience();
	affichage_courbe(proba_position);

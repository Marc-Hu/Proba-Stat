import email
import re
import matplotlib.pyplot as plt
import numpy as np
import operator

import time
import copy
import sys
import itertools

import nltk
from nltk.stem import PorterStemmer

GROUPEMENT = 10;
LIMIT = 12;

##
#  Récupérer les mails
##
def read_file(fname):
    """ Lit un fichier compose d'une liste de emails, chacun separe par au moins 2 lignes vides."""
    f = open(fname,'rb')
    raw_file = f.read()
    f.close()
    raw_file = raw_file.replace(b'\r\n',b'\n')
    emails =raw_file.split(b"\n\n\nFrom")
    emails = [emails[0]]+ [b"From"+x for x in emails[1:] ]
    return emails

def get_body(em):
    """ Recupere le corps principal de l'email """
    body = em.get_payload()
    if type(body) == list:
        body = body[0].get_payload()
    try:
        res = str(body)
    except Exception:
        res=""
    return res

def clean_body(s):
    """ Enleve toutes les balises html et tous les caracteres qui ne sont pas des lettres """
    patbal = re.compile('<.*?>',flags = re.S)
    patspace = re.compile('\W+',flags = re.S)
    return re.sub(patspace,' ',re.sub(patbal,'',s))

def get_emails_from_file(f):
    mails = read_file(f)
    return [ s for s in [clean_body(get_body(email.message_from_bytes(x))) for x in mails] if s !=""]

##
#   Exo 1
##
def split(liste, x):
	listeA=[];
	leng=len(liste);
	for i in range(int(leng*x/100)):
		listeA.append(liste.pop(0));
	return listeA, liste;

##
#   Exo 2
##

def len_email(liste):
    len_liste = [];
    for i in range (len(liste)):
        len_liste.append(len(liste[i].split()));
        # print(liste[i] ,len(liste[i]));
    return len_liste;

# Fonction qui affiche l'histogramme des mails d'apprentissage spam et nospam
def affiche_history(spam, nospam):
    # print(spam, nospam);
    plt.hist(spam, bins=300, color = 'burlywood', label = "Longueur des emails spam", fc=(0, 0, 1, 0.5))
    plt.hist(nospam, bins=300, color = 'yellow', label = "Longueur des emails nospam", fc=(1, 0, 0, 0.5))
    plt.xlabel("longueur")
    plt.ylabel("Nombre d'emails")
    plt.title('Nombre d\'emails en fonction de la longueur')
    plt.legend()
    plt.gca().set_xlim([0,500]);
    plt.show()

# Fonction qui va renvoyer la valeur qui limitera la classification
# Fonction pour la première moitié d'une liste de nb de mail
# Exemple : On veut une classification entre 10 lignes (compris) et 30 lignes (non compris)
# Si 30 n'apparait pas dans la liste des nb de mot des mails (mails) alors on décrémente
# On essaye de trouver la valeur la plus proche par rapport à 30
# On initialise k à 1 car on ne veut pas 30 (non compris)
def getIndexLimit3(limits, mails):
    k=1;
    while limits[3]==-1 :# Tant qu'on a pas trouvé
        try :
            limits[3] = mails.index(int(limits[1]) - k); #On essaye de trouver l'index
            if mails[limits[3]+1] == mails[limits[3]] :# Si sa fonctionne on regarde si la valeur à droite n'est pas la même
            #En effet, mails.index(x) renvoi la première occurence, or il se peut qu'il y ait plusieurs valeurs x cote à cote
                hasDuplicate = True;
                while hasDuplicate : #Tant qu'on à une même valeur
                    if(mails[limits[3]+1] != mails[limits[3]]) :
                        hasDuplicate=False; #Retourne false si la valeur à indice+1 n'est plus la même
                    else :
                        limits[3]=limits[3]+1; # Sinon on incrémente la l'indice
        except :
            k=k+1; #Si la recherche d'index echoue, on incrémente k
    return limits[3]; 

# Fonction qui ressemble à celle du haut sauf que c'est pour la limite à gauche
# Fonction pour la première moitié d'une liste de nb de mail
# Exemple : Classification entre 10 lignes (compris) et 20 lignes (non compris)
# Si 10 n'apparaît pas dans la liste alors on cherche avec 10+i
# On initialise i à 0 car 10 est compris dans l'intervalle
def getIndexLimit2(limits, mails, current_value):
    i=0;
    while limits[2]==-1 : #Tant qu'on ne trouve pas l'indice 
        if current_value-GROUPEMENT<0 : #Si on arrive à une valeur négative
            limits[2]=0; #Alors l'indice est forcément 0 car on arrive à la limite
        else : #Sinon
            try :
                limits[2] = mails.index(current_value - GROUPEMENT + i); #On essaye de trouver l'indice
            except :
                i=i+1; #Sinon on incrémente i
    return limits[2];

# Fonction qui va renvoyer la première moitié d'un modele de mail (spam ou non spam)
def first_half(mails, limits, current_value, mails_len):
    j = 0;
    result = {};
    while j<=LIMIT and limits[0] != 0 : #On limite le nombre d'interval
        if j<LIMIT : #Si la limite n'est pas encore atteinte
            limits[2]=-1;
            limits[2]=getIndexLimit2(limits, mails, current_value); #On cherche la limite à gauche
            limits[3]=getIndexLimit3(limits, mails); #Puis à droite de l'interval
        else : #Si c'est la limite
            limits[2] = 0; #La valeur la plus à gauche est forcément 0
            limits[0] = 0; # L'indice la plus à gauche est forcément 0
            limits[3]=getIndexLimit3(limits, mails);
        # On rajoute 1 à la valeur à droite (limits[3]+1) car on doit aussi récupérer la valeur à cette indice
        array = mails[limits[2] : limits[3]+1]; # On récupère l'intervalle de valeur grâce au limite d'indice trouvé précédement
        # print(mails[limits[2] : limits[3]+1]);
        result[limits[0]]=len(array)/mails_len; # On fait la moyenne par rapport au nb de mail testé
        if(j<LIMIT): 
            limits[0]=limits[0]-GROUPEMENT; #On décrémente la valeur de la limite gauche
            limits[1]=limits[1]-GROUPEMENT; #De même pour la valeur à droite
            current_value=limits[1]; 
            limits[2]=-1; #On réinitialise les indices 
            limits[3]=-1;
        j=j+1;
    return result;

# Même fonction que celui au dessus mais pour la deuxième partie de la liste
def getIndexLimit2ForSecondHalf(limits, mails, current_value):
    i=0;
    while limits[2]==-1 :
        try :
            limits[2] = mails.index(current_value + i);
        except :
            i=i+1;
    return limits[2];

# Même fonction que celui au dessus mais pour la deuxième partie de la liste
def getIndexLimit3ForSecondHalf(limits, mails):
    k=1; #On initialise à 1 car on ne prend pas les valeurs tout à droite (non comprise)
    while limits[3]==-1 :
        try :
            limits[3] = mails.index(int(limits[1]) - k);
            if mails[limits[3]+1] == mails[limits[3]] :
                hasDuplicate = True;
                while hasDuplicate :
                    if(mails[limits[3]+1] != mails[limits[3]]) :
                        hasDuplicate=False;
                    else :
                        limits[3]=limits[3]+1;
        except :
            k=k+1;
    return limits[3];

# Même fonction que pour first_half mais cette fois-ci c'est pour la deuxième partie
def second_half(mails, limits, current_value, mails_len):
    j = 0;
    result = {};
    # print(limits);
    limits[0]=current_value;
    limits[1]=current_value+GROUPEMENT;
    while j<=LIMIT and limits[3] != len(mails)-1:
        if j<LIMIT :
            limits[2]=-1;
            limits[3]=-1;
            limits[2]=getIndexLimit2ForSecondHalf(limits, mails, current_value);
            limits[3]=getIndexLimit3ForSecondHalf(limits, mails);
        else :
            # limits[0] = mails[len(mails)-1];
            limits[2] = getIndexLimit2ForSecondHalf(limits, mails, current_value);
            limits[3] = len(mails)-1;
        array = mails[limits[2] : limits[3]+1];
        # print(mails[limits[2] : limits[3]+1]);
        # print(result);
        result[limits[0]]=len(array)/mails_len;
        if j<LIMIT :
            limits[0]=limits[0]+GROUPEMENT;
            limits[1]=limits[1]+GROUPEMENT;
            current_value=limits[0];
            limits[2]=-1;
            limits[3]=-1;
        j=j+1;
    return result;

def get_mail_modele(mails, mediane_ref):
    current_value = -1;
    i=0;
    while current_value==-1:
        try:
            res = mails.index(mediane_ref-i) 
            current_value=mails[res];
        except :
            i=i+1;
    mails_len = len(mails);
    result = {}; #Stock des résultat
    # limits[0]=valeur limite gauche (comprise); limits[1]=valeur limite droite (non comprise)
    # limits[2]=indice limite gauche; limits[3]=indice limite droite
    limits=[mediane_ref-GROUPEMENT, current_value, 0,mails.index(current_value)-1]; #Init les limites
    result = first_half(mails, limits, current_value, mails_len); #Première partie
    # print(result);
    current_value=mediane_ref;
    result2 = second_half(mails, limits, current_value, mails_len); #Deuxième partie
    for key, value in result2.items(): #On va fusionner la deuxième partie avec la première
        result[key]=value;
    val=0;

    # for key, value in result.items(): #check si la somme fait bien 1 ou très proche de 1 car il y a des arrondissements lors des calculs des proba
    #     val=val+value;
    # print("test bien à 1", val);
    return result;

#Fonction apprend modèle qui va renvoyer un tableau de tuple
#Chaque tuple est représenté par : (nb de mots dans un mail, proba que sa soit un spam)
def apprend_modele(spam1, nospam1, mediane_ref):
    spam_mail_model=get_mail_modele(spam1, mediane_ref);
    no_spam_mail_model=get_mail_modele(nospam1, mediane_ref);
    spam_mail_model=sorted(spam_mail_model.items());
    no_spam_mail_model=sorted(no_spam_mail_model.items());
    result=list();
    # print(spam_mail_model); #Affiche le groupement
    for i in range (len(spam_mail_model)):
        result.append((spam_mail_model[i][0], spam_mail_model[i][1]/(spam_mail_model[i][1]+no_spam_mail_model[i][1])));
    return result;

# Fonction qui va prédire pour un modèle donnée, si les mails de test sont des spams ou non (respectivement 1 ou -1) 
def predit_email(mails, model) : 
    # print(model);
    result = list();
    for i in range(len(mails)):
        mail_len=len(mails[i].split());
        for j in range (len(model)): # On cherche dans quel intervalle ce situe le mail actuel
            if mail_len<model[j][0] :
                if model[j-1][1]>0.5 : # Si la proba que sa soit un spam est strictement supérieur à 0.5
                    result.append((i, 1)); # Alors le mail est un spam
                else :
                    result.append((i, -1)); # Sinon ce n'est pas un spam
                break;
    return result;


def estimation_erreur(spam, nospam):
    list_x = [50, 60, 70, 80, 90]
    list_erreur = list()
    percentage_error=[];
    for x in list_x:
        # print(len(spam), len(nospam));
        spam_copy=copy.copy(spam); # Copie de la liste des spams pour éviter de perdre la liste initiale
        nospam_copy=copy.copy(nospam);
        set_train_spam, set_test_spam = split(spam_copy, x)
        set_train_nospam, set_test_nospam = split(nospam_copy, x)
        spam1 = sorted(len_email(set_train_spam));
        nospam1 = sorted(len_email(set_train_nospam));
        # affiche_history(spam1, nospam1)
        mediane_ref = spam1[int(len(spam1)/2)]; #Prendre la médiane
        modele = apprend_modele(spam1, nospam1, mediane_ref)
        predict_spam = predit_email(set_test_spam, modele)
        predict_nospam = predit_email(set_test_nospam, modele)
        cpt = 0
        for i in range(len(predict_spam)):
            if(predict_spam[i][1] != 1):
                cpt += 1
        for i in range(len(predict_spam)):
            if(predict_nospam[i][1] != -1):
                cpt += 1
        cpt = float(cpt) / (len(predict_spam)+len(predict_nospam))
        percentage_error.append(cpt);
        print("Pourcentage d'erreur : ", cpt, " pour : ", x, "pourcent d'exemples");
        list_erreur.append(cpt)
    # print(list_erreur)
    axis=[40, 100, 0, 1];
    title = "Variation du taux d'erreur par rapport à la taille de l'ensemble d'apprentissage";
    xlabel="Pourcentage d'exemples";
    # affiche_pourcent_erreur(list_x, list_erreur, axis, title, xlabel);
    return percentage_error;

def affiche_pourcent_erreur(list_x, list_erreur, axis, title, xlabel) :
    plt.plot(list_x, list_erreur, 'ro')
    plt.axis(axis)
    plt.xlabel(xlabel)
    plt.ylabel("Taux d'erreur")
    plt.title(title)
    plt.legend()
    plt.show()

##
#   Exo 3
##

# Fonction qui va réduire le nombre de mot dans notre dictionnaire
def reduce_dictionary(dictionnary, min_val_tab):
    dictionary_leaned1={};
    total_word=0;
    result=list();
    for min_val in min_val_tab :
        for key, value in dictionnary.items():
            if value>min_val and value<300 and len(key)<28 and re.search("^[a-zA-Z]*$", key): #Si le mot est pas trop long
                dictionary_leaned1[key]=value;
                total_word=total_word+value;
        result.append((dictionary_leaned1, total_word));
        dictionary_leaned1={};
        total_word=0;
    return result;

#Fonction qui va renvoyer un dictionnaire de mots qui sont dans un mail spam
def get_dictionary(mails, min_val_tab):
    # print(len(spam), len(nospam));
    copy_mails=copy.copy(mails);
    start = time.clock();
    stemmer = PorterStemmer(); #Stemmer pour prendre la racine d'un mot
    result={};
    for i in range (len(copy_mails)): #Pour tous les mails spams
        mail = copy_mails[i].split(); #on sépare le mails par ses mots
        # print(mail);
        for j in range(len(mail)): #Pour tout ces mots
            if stemmer.stem(mail[j].lower()) in result : #On incrémente si il est déjà dans le dictionnaire
                result[stemmer.stem(mail[j].lower())]=result[stemmer.stem(mail[j].lower())]+1;
            else : #On l'ajoute sinon
                result[stemmer.stem(mail[j].lower())]=1;
    res=reduce_dictionary(result, min_val_tab); #res contient un tuple (dictionnaire, nb mot dans le dictionnaire)
    end = time.clock();
    # print(res[0] , "\nNombre de mots différent dans le dictionnaire : ", len(res[0]), "\nTemps d'exécution de la fonction : ", end-start, "s.\nNombre total de mot dans le dictionnaire : ", res[1]);
    return res;

# Fonction qui va renvoyer en pourcentage, le nombre d'occurence d'un mot dans un dictionnaire
# L'argument dictionnaire contient un tuple (dictionnaire, nb mot dans le dico)
def occu_word(word, dictionnaire):
    stemmer = PorterStemmer();
    if stemmer.stem(word) in dictionnaire[0] :
        res = np.log(dictionnaire[0][stemmer.stem(word)]/float(dictionnaire[1]));
        return res;
    return 0;

def proba_conditionelle_email(body, dictionnaire):
    result=0;
    for word in body.split() :
        result=result+occu_word(word, dictionnaire);
    return result;

def classifieur(emails, nb_spam_training, nb_nospam_training, dictionnaire_spam, dictionnaire_nospam):
    result=list();
    nb_spam=0;
    for mail in emails :
        est_spam = nb_spam_training * proba_conditionelle_email(mail, dictionnaire_spam);
        pas_spam = nb_nospam_training * proba_conditionelle_email(mail, dictionnaire_nospam);
        result.append(est_spam<pas_spam);
        if est_spam<pas_spam :
            nb_spam=nb_spam+1;
    return (result, nb_spam);

# Fonction qui va estimer les erreurs du classifieur bayésien naïve en réduisant pas à pas la taille du dictionnaire
def estimation_erreur_classifieur(spam, nospam) :
    list_x = [10];
    list_nb_word=[];
    list_erreur = list();
    copy_spam = copy.copy(spam); #Faire des copies pour éviter que l'original ne soit écrasé
    copy_nospam = copy.copy(nospam);
    apprentice_spam, test_spam=split(copy_spam, 80); #Split les mails en mails d'apprentissage et mail de test
    apprentice_nospam, test_nospam = split(copy_nospam, 80);
    nb_spam_training=len(apprentice_spam)/len(apprentice_spam)+len(apprentice_nospam);
    nb_nospam_training=len(apprentice_nospam)/len(apprentice_spam)+len(apprentice_nospam);
    dictionnaire_spam = get_dictionary(apprentice_spam, list_x); #Récupérer le dictionnaire des spams
    dictionnaire_nospam = get_dictionary(apprentice_nospam, list_x);
    for i in range (len(dictionnaire_spam)) :
        result_spam=classifieur(test_spam, nb_spam_training, nb_nospam_training, dictionnaire_spam[i], dictionnaire_nospam[i])
        result_nospam=classifieur(test_nospam, nb_spam_training, nb_nospam_training, dictionnaire_spam[i], dictionnaire_nospam[i])
        erreur = float(len(result_spam[0])-result_spam[1]+result_nospam[1]) / (len(result_spam[0])+len(result_nospam[0]));
        list_erreur.append(erreur);
        list_nb_word.append(len(dictionnaire_spam[i][0]))
        # print(list_nb_word);
    # print("Pourcentage d'erreur minimum : "+str(int(list_erreur[0]*100)), "%");
    # axis=[1500, 9000, 0, 1];
    # title="Variation du taux d'erreur par rapport à la réduction du dictionnaire"
    # xlabel="Nombre de mot dans le dictionnaire"
    # affiche_pourcent_erreur(list_nb_word, list_erreur, axis, title, xlabel);
    return dictionnaire_spam, dictionnaire_nospam;

##
#   2.Visualisation
##

# #Calculer la scale_distance euclidienne entre l'indiv et le reste de la population
# def scale_distance_Euclidienne(indiv , Data, sigma):
#     return np.divide(np.sqrt(np.sum((indiv-Data)**2, axis=1)),2*sigma)

# #Calculer la distance euclidienne entre l'indiv et le reste de la population
# def distance_Euclidienne(indiv , Data):
#     return np.sqrt(np.sum((indiv-Data)**2, axis=1))

# #calculer la matrice corespendant à la formule exp -d(xi,xj)/2*sigma
# def calcul_matrice_Distance_Exp(data, sigma):
#     nbE = data.shape[0]
#     #Calculer les distances entre les individus 
#     MDistance = np.zeros(shape=(nbE,nbE))
#     for i, indiv in zip(range(0,nbE-1,1),data): #le dernier individu on ne le prend pas en considération
#         RowDi = scale_distance_Euclidienne(indiv, data[i+1:nbE,:], sigma)
#         #print(RowDi)
#         MDistance[i,(i+1):nbE] = RowDi
#         MDistance[(i+1):nbE, i] = RowDi
#     return np.exp((-1)*MDistance)

# def calcul_Pij(MDistance):
#     nbE = MDistance.shape[0]
#     Pij = np.zeros(shape=(nbE,nbE))
#     for i in range(0,nbE,1):
#         for j in range(0,nbE,1):
#             Pij[i,j] = MDistance[i,j] / (np.sum(MDistance[i,:])-MDistance[i,i])
#     return Pij

# def calcul_Qij(MDistanceY):
#     nbE = MDistanceY.shape[0]
#     Qij = np.zeros(shape=(nbE,nbE))
#     for i in range(0,nbE,1):
#         for j in range(0,nbE,1):
#             Qij[i,j] = MDistanceY[i,j] / (np.sum(MDistanceY[i,:])-MDistanceY[i,i])
#     return Qij


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


def email_vect(email, collection):
    stemmer = PorterStemmer();
    """ Fonction pour représenter un email saus la forme d'un vecteur selon un vocabulaire donné """
    words_email = [stemmer.stem(word) for word in email.split()] # Utiliser un stemmer pour réduire le nombre de mots
    binary_rep = [];
    b = False
    # print(collection);
    for key, value in collection.items():
        for i in words_email:
            if(key == i):
                b = True
                break
        if(b):
            binary_rep.append(1)
        else:
            binary_rep.append(0)
        b = False
    return binary_rep

def random_y(emails, dictionnary):
    result=[];
    for key, value in dictionnary.items():
        result.append(np.random.normal(0, 0.5));
    # for i in range (len(emails)):
    #     result.append(np.random.normal(0, 0.5));
    return result;

#Fonction qui va vectoriser tous les mails spam et non spam selon un dictionnaire
def vectorize_emails(spam, nospam):
    collection_spam, collection_nospam=estimation_erreur_classifieur(spam, nospam);
    mails_spam, test_mails_spam=split(spam, 80)
    mails_nospam, test_mails_nospam=split(nospam, 80)
    test_mails_nospam=test_mails_nospam[0:len(test_mails_spam)]
    spam_email_vec=[];
    nospam_email_vec=[];
    random_y1=[];
    random_y2=[];
    y = np.random.normal(0,0.5, size=(len(test_mails_spam),2))
    for mail in test_mails_spam :
        spam_email_vec.append(email_vect(mail, collection_nospam[0][0]));
        random_y1.append(random_y(mail, collection_nospam[0][0]));
    for mail in test_mails_nospam :
        nospam_email_vec.append(email_vect(mail, collection_nospam[0][0]));
        random_y2.append(random_y(mail, collection_nospam[0][0]));
    # print(spam_email_vec, nospam_email_vec);
    print("pij1");
    # print(spam_email_vec);
    res_proba_pij1 = proba_pij_qij(spam_email_vec, True); # Récupère les proba pij
    # print(res_proba_pij1);
    print("pij2");
    res_proba_pij2 = proba_pij_qij(nospam_email_vec, True);
    # random_y1 = [random_y(spam_email_vec, collection_spam[0][0])];
    # print(random_y1);
    # random_y2 = [random_y(nospam_email_vec, collection_spam[0][0])];
    print("qij1");
    res_proba_qij1 = proba_pij_qij(random_y1, False); #Récupère les proba qij
    print("qij2");
    res_proba_qij2 = proba_pij_qij(random_y2, False);
    # print(res_proba_qij1)
    return res_proba_pij1, res_proba_pij2, res_proba_qij1, res_proba_qij2, y;

def proba_pij_qij(emails_vectorized, bool_for_pij):
    result=np.empty((len(emails_vectorized), len(emails_vectorized)));
    # result[0, 0]=0.01
    # print(result);
    eucli_result=[];
    j=0;
    for i in range(len(emails_vectorized)) :
        while j<len(emails_vectorized)+i:
            # if(j==i) :
            #     j=j+1;
            if(j>=len(emails_vectorized)+i):
                break;
            res=scale_squared_euclidian_distance(emails_vectorized[i], emails_vectorized[j%len(emails_vectorized)], len(emails_vectorized), bool_for_pij);
            res=np.exp(-res)
            eucli_result.append(res);
            # print(res, i, j, len(emails_vectorized));
            j=j+1;
        for k in range(len(eucli_result)):
            # print(np.sum(eucli_result[0 : len(eucli_result)])-eucli_result[k])
            result[i, k]=eucli_result[k]/(np.sum(eucli_result[0 : len(eucli_result)])-eucli_result[k])
        # result.append(eucli_result[0]/np.sum(eucli_result[1:len(eucli_result)]));
        eucli_result=[];
        j=i+1;
    # print(result);
    return result;

def scale_squared_euclidian_distance(vect_i, vect_j, len_neighbors, bool_for_pij):
    # print(vect_i);
    result=[];
    res=0.0;
    for i in range (len(vect_i)):
        res=res+float(vect_i[i]*vect_i[i]-vect_j[i]*vect_j[i])
    # print(res);
    res=res*res;
    res=float(np.sqrt(res));
    # print(res, float(2*(np.square(np.log(len_neighbors)))));
    if bool_for_pij :
        res=res/float(2*(np.square(np.log(len_neighbors))));
    return res;

def sne_algo(spam, nospam):
    result = vectorize_emails(spam, nospam); # Result = [pij1, pij2, qij1, qij2, y]
    print(result);

    for i in range(0,25):
        y=calcul_GradiantY(result[4],result[0],result[2],0)
        x=calcul_GradiantY(result[4],result[1],result[3],0)

    print('Y apres 100 intération')
    # #PS : pour le plot et l'affichage a la fin une fois que vous avez les Y labelisez les selon le label des données initiales
    # #genre par exemple si vous avez 100mail non spam il corresponde aux 100 premier ligne de Y et vice versa
    print(y)
    plt.scatter(y[:,0], y[:,1], c = 'red');
    plt.scatter(x[:,0], x[:,1], c = 'blue');
    plt.legend(["spam","nospam"])
    plt.title("Nuage de points pour le dictionnaire des non spams")
    plt.show();


##
#   Main
##


if __name__ == '__main__':
    spam = get_emails_from_file("spam.txt" )
    nospam = get_emails_from_file("nospam.txt");
    ######
    # EXO 2
    ######
    if sys.argv[1]=="exo2":
        result=estimation_erreur(spam, nospam);
        print(np.sum(result)/len(result));
    ######
    # EXO 3
    ######
    if sys.argv[1]=="exo3":
        estimation_erreur_classifieur(spam, nospam);
    ######
    # EXO VISUALISATION
    ######
    if sys.argv[1]=="sne" :
        start = time.clock();
        res = sne_algo(spam, nospam);
        # for r in res :
        #     print(r, "\n");
        print(time.clock()-start, "secondes.");

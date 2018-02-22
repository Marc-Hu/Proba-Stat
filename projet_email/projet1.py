import email
import re
import matplotlib.pyplot as plt
import numpy
import operator
import nltk
import time
import copy
import sys
from nltk.stem import PorterStemmer
import langdetect
from langdetect import detect

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

# Fonction qui affiche l'historigramme des mails d'apprentissage spam et nospam
def affiche_history(spam, nospam):
    plt.legend(loc='upper right');
    plt.hist(spam, bins=200, normed=1);
    plt.hist(nospam, bins=200, normed=1, histtype='bar');
    plt.gca().set_xlim([0,1500]);
    plt.show();

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
    current_value=mediane_ref;
    result2 = second_half(mails, limits, current_value, mails_len); #Deuxième partie
    for key, value in result2.items(): #On va fusionner la deuxième partie avec la première
        result[key]=value;
    val=0;
    # for key, value in result.items(): #check si la somme fait bien 1 ou très proche de 1 car il y a des arrondissements lors des calculs des proba
    #     val=val+value;
    # print("test bien à 1", val);
    result = sorted(result.items(), key=operator.itemgetter(0)) # On trie
    # print(result);
    return result;

#Fonction apprend modèle qui va renvoyer un tableau de tuple
#Chaque tuple est représenté par : (nb de mots dans un mail, proba que sa soit un spam)
def apprend_modele(spam1, nospam1, mediane_ref):
    spam_mail_model=get_mail_modele(spam1, mediane_ref);
    no_spam_mail_model=get_mail_modele(nospam1, mediane_ref);
    result=list();
    # print(spam_mail_model, no_spam_mail_model); #Affiche le groupement
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
    for x in list_x:
        # print(len(spam), len(nospam));
        spam_copy=copy.copy(spam); # Copie de la liste des spams pour éviter de perdre la liste initiale
        nospam_copy=copy.copy(nospam);
        set_train_spam, set_test_spam = split(spam_copy, x)
        set_train_nospam, set_test_nospam = split(nospam_copy, x)
        spam1 = sorted(len_email(set_train_spam));
        nospam1 = sorted(len_email(set_train_nospam));
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
        print("Pourcentage d'erreur : ", cpt, " pour : ", x, "pourcent d'exemples");
        list_erreur.append(cpt)
    # print(list_erreur)
    plt.plot(list_x, list_erreur, 'ro')
    plt.axis([40, 100, 0, 1])
    plt.xlabel("Pourcentage d'exemples")
    plt.ylabel("Taux d'erreur")
    plt.title("Variation du taux d'erreur par rapport à la taille de l'ensemble d'apprentissage")
    plt.legend()
    plt.show()


##
#   Exo 3
##


# Fonction qui va réduire le nombre de mot dans notre dictionnaire
def reduce_dictionary(dictionnary):
    dictionary_leaned1={};
    dictionary_leaned2={};
    total_word=0;
    i=0;
    for key, value in dictionnary.items():
        if value>5 and len(key)<28 and re.search("^[a-zA-Z]*$", key): #Si le mot est pas trop long
            dictionary_leaned1[key]=value;
            total_word=total_word+value;
    # La boucle en dessous permet de reduire le dictionnaire en prenant que les mots en anglais (mais plus assez de mot dans le dico si on l'applique)
    #print(detect("5th"));
    # for key, value in dictionary_leaned1.items():    
    #     try:
    #         if detect(key)=="en" :
    #             print(key, i);
    #             dictionary_leaned2[key]=value;
    #             total_word=total_word+value;
    #     except :
    #         pass
    #     i=i+1;
    return dictionary_leaned1, total_word;

#Fonction qui va renvoyer un dictionnaire de mots qui sont dans un mail spam
def get_dictionary(spam, nospam):
    # print(len(spam), len(nospam));
    start = time.clock();
    stemmer = PorterStemmer(); #Stemmer pour prendre la racine d'un mot
    result={};
    result_leaned={};
    for i in range (len(spam)): #Pour tous les mails spams
        mail = spam[i].split(); #on sépare le mails par ses mots
        # print(mail);
        for j in range(len(mail)): #Pour tout ces mots
            if stemmer.stem(mail[j].lower()) in result : #On incrémente si il est déjà dans le dictionnaire
                result[stemmer.stem(mail[j].lower())]=result[stemmer.stem(mail[j].lower())]+1;
            else : #On l'ajoute sinon
                result[stemmer.stem(mail[j].lower())]=1;
    for i in range (len(nospam)): #Pour tous les mails non spam d'apprentissage
        mail = nospam[i].split();
        for j in range(len(mail)):
            if stemmer.stem(mail[j].lower()) in result :
                del result[stemmer.stem(mail[j].lower())]; #Si le mot existe dans le dictionnaire alors on le supprime de la liste
                # result[stemmer.stem(mail[j])]=result[stemmer.stem(mail[j])]-1;
                # if result[stemmer.stem(mail[j])]==0 :
                #     del result[stemmer.stem(mail[j])];
    result, total_word=reduce_dictionary(result);
    end = time.clock();
    print(result , "\nNombre de mots différent dans le dictionnaire : ", len(result), "\nTemps d'exécution de la fonction : ", end-start, "s.\nNombre total de mot dans le dictionnaire : ", total_word);
    return result;

def email_vect(email, collection):
    stemmer = PorterStemmer();
    """ Fonction pour représenter un email saus la forme d'un vecteur selon un vocabulaire donné """
    words_email = [stemmer.stem(word) for word in email.split()] # Utiliser un stemmer pour réduire le nombre de mots
    binary_rep = {}
    b = False
    for key, value in collection.items():
        for i in words_email:
            if(key == i):
                b = True
                break
        if(b):
            binary_rep[key] = 1
        else:
            binary_rep[key] = 0
        b = False
    return binary_rep

# def occur_word(emails):
#     result = {}
#     for email in emails:
#         # Utilisation d'un stemmer
#         words_email = [word for word in email.split()]
#         for word in words_email:
#             # Ajout des mots de l'email au dictionnaire des résultats
#             if word not in result:
#                 result[word] = 1 
#             # Parcourir les autres emails pour calculer le nombre d'occurrence des mots contenus dans le mail
#             for mail in emails:
#                 if email != mail:
#                     br = email_vect(mail, words_email)
#                     for w in words_email:
#                         if br[w] == 1:
#                             result[word] += 1
#     return result

def hist_occur(spam, nospam):
    r = get_dictionary(spam, nospam)
    # keys = list(r.keys()) #[k for k in r.keys()]
    # values = list(r.values())
    # plt.bar(list(range(len(keys))), values, color = 'blue')
    # plt.xlabel("Mots")
    # plt.ylabel("Nombre d'apparitions")
    # plt.title("Nombre d'apparitions des mots dans les emails")
    # plt.legend()
    # plt.show()
    return r;

def predict_email_with_dict(spam, nospam, dictionnaire) :
    stemmer = PorterStemmer();
    nb_predict_spam=0;
    nb_predict_nospam=0;
    error=[];
    found_spam=False;
    for i in range (len(spam)) :
        words_email = [stemmer.stem(word) for word in spam[i].split()];
        found_spam=False;
        for j in range(len(words_email)):
            for key, value in dictionnaire.items():
                if key == words_email[j] :
                    nb_predict_spam=nb_predict_spam+1;
                    found_spam=True;
                    break;
            if found_spam :
                break;
    error.append(1-(nb_predict_spam/len(spam)));
    # nb_predict_spam=0;
    # found_spam=False;
    # for i in range (len(nospam)) :
    #     words_email = [stemmer.stem(word) for word in nospam[i].split()];
    #     found_spam=False;
    #     for j in range(len(words_email)):
    #         for key, value in dictionnaire.items():
    #             if key == words_email[j] :
    #                 nb_predict_spam=nb_predict_spam+1;
    #                 found_spam=True;
    #                 break;
    #         if found_spam :
    #             break;
    # error.append(nb_predict_spam/len(spam));
    # print(error);
    return error;


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
        estimation_erreur(spam, nospam);
    ######
    # EXO 3
    ######
    if sys.argv[1]=="exo3":
        apprentice_spam, test_spam=split(spam, 80);
        apprentice_nospam, test_nospam = split(nospam, 80);
        spam1 = sorted(len_email(apprentice_spam));
        nospam1 = sorted(len_email(apprentice_nospam));
        dictionnaire = hist_occur(apprentice_spam, apprentice_nospam);
        print(predict_email_with_dict(test_spam, test_nospam, dictionnaire));
        print(predict_email_with_dict(apprentice_spam, apprentice_nospam, dictionnaire));
        # print();
        # email_vect(test_nospam[0], dictionnaire)

import email
import re
import matplotlib.pyplot as plt
import numpy
import operator

GROUPEMENT = 150;
LIMIT = 5;
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

def split(liste, x):
	listeA=[];
	leng=len(liste);
	for i in range(int(leng*x/100)):
		listeA.append(liste.pop(0));
	return listeA, liste;

def len_email(liste):
	len_liste = [];
	for i in range (len(liste)):
		len_liste.append(len(liste[i]));
	return len_liste;

# Fonction qui affiche l'historigramme des mails d'apprentissage spam et nospam
def affiche_history(spam, nospam):
    plt.legend(loc='upper right');
    plt.hist(spam, bins=400, normed=1);
    plt.hist(nospam, bins=400, normed=1, histtype='bar');
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

# Fonction qui va renvoyer le modele par rapport à une liste des nb de mots dans un mails (spam ou nospam)
# La valeur mediane_ref permet d'avoir la même valeur de la médiane pour un nombre x de mails spam et y de mais non spam
def apprend_modele(mails, mediane_ref):
    current_value = mediane_ref;
    mails_len = len(mails);
    result = {}; #Stock des résultat
    # limits[0]=valeur limite gauche (comprise); limits[1]=valeur limite droite (non comprise)
    # limits[2]=indice limite gauche; limits[3]=indice limite droite
    limits=[mediane_ref-GROUPEMENT, mediane_ref, 0,mails.index(current_value)-1]; #Init les limites
    result = first_half(mails, limits, current_value, mails_len); #Première partie
    current_value=mediane_ref;
    result2 = second_half(mails, limits, current_value, mails_len); #Deuxième partie
    for key, value in result2.items(): #On va fusionner la deuxième partie avec la première
        result[key]=value;
    val=0;
    # for key, value in result.items(): #check si la somme fait bien 1 ou très proche de 1 car il y a des arrondissements lors des calculs des proba
    #     val=val+value;
    # print(val);
    result = sorted(result.items(), key=operator.itemgetter(0)) # On trie
    print(result);
    return result;

# Fonction qui va prédire si un mail est un spam ou non par rapport à un modèle
def predit_email(mails, model) : #model[0]=spam; model[1]=nospam
    result=list();
    found_position = False;
    index = 0;
    for i in range(len(mails)): # Pour tous les mails
        found_position=False;
        index=0;
        while not found_position : # On essaye de trouver la position de la valeur dans le modèle
            # print(index);
            if model[0][index][0]<mails[i] : # Si la valeur est supérieur à la valeur d'index 'index'
                index=index+1; # On incrément pour voir la prochaine valeur
                if index>=len(model[0]) : #Si on a atteint la limite, sa veut dire que la valeur se trouve dans la partie la plus à droite
                    index = len(model[0])-1;
                    found_position=True;
            elif model[0][index][0]==mails[i] :
                found_position=True;
            elif model[0][index][0]>mails[i] : # Si la valeur est inférieur alors on se trouve dans l'intervalle (index-1 et index)
                index=index-1;
                found_position=True;
        # print(model[0][index][1], model[1][index][1])
        if model[0][index][1]>model[1][index][1] : #Si il a plus de chance d'être un spam
            result.append((mails[i], True)); #Alors on rajoute en tant que spam (True)
        else :
            result.append((mails[i], False)); # False sinon
    return result;


if __name__ == '__main__':
    spam = get_emails_from_file("spam.txt" )
    nospam = get_emails_from_file("nospam.txt");
    listeA, listeB = split(spam, 10);
    spam1 = sorted(len_email(spam));
    nospam1 = sorted(len_email(nospam));
    # print(spam1[int(len(spam1)/2)], nospam1[int(len(nospam1)/2)], spam1);
    # affiche_history(spam1, spam2);
    mediane_ref = spam1[int(len(spam1)/2)];
    spam_model = apprend_modele(spam1, mediane_ref);
    noSpam_model = apprend_modele(nospam1, mediane_ref)
    model = (spam_model, noSpam_model);
    print(model);
    print(predit_email(spam1, model));
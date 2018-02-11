import email
import re
import matplotlib.pyplot as plt
import numpy
import operator

GROUPEMENT = 150;
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

def affiche_history(spam, nospam):
    # bins = numpy.linspace(0, 1500);
    # plt.hist(spam, bins, alpha=0.5, label='spam');
    # plt.hist(nospam, bins, alpha=0.5, label='no spam');
    plt.legend(loc='upper right');
    # plt.show();
    plt.hist(spam, bins=400, normed=1);
    plt.hist(nospam, bins=400, normed=1, histtype='bar');
    plt.show();

def apprend_modele(mails):
    mediane_mails = mails[int(len(mails)/2)];
    current_value = mediane_mails;
    limitB = mails.index(current_value);
    mails_len = len(mails);
    result = {};
    limit = 5;
    j = 0;
    # print(mediane_spam, mediane_nospam);
    while j<=limit :
        if j<limit :
            limitA=-1;
            i=0;
            while limitA==-1 :
                if current_value-GROUPEMENT<0 :
                    limitA=0;
                else :
                    try :
                        limitA = mails.index(current_value - GROUPEMENT - i);
                    except :
                        i=i+1;
        else :
            limitA = 0;
        array = mails[limitA : limitB];
        result[limitA]=len(array)/mails_len;
        current_value=mails[limitA];
        limitB=limitA;
        j=j+1;
    limitA=mails.index(mediane_mails);
    limitB=-1;
    current_value=mediane_mails;
    j=0;
    while j<=limit :
        if j<limit :
            limitB=-1;
            i=0;
            while limitB==-1 :
                if current_value+GROUPEMENT>mails[len(mails)-1] :
                    limitB = len(mails)-1;
                else :
                    try :
                        limitB = mails.index(current_value + GROUPEMENT + i);
                    except :
                        i=i+1;
                    # print(limitA, limitB);
        else :
            limitB = len(mails)-1;
        array = mails[limitA : limitB];
        # print(len(array));
        result[limitA]=len(array)/mails_len;
        current_value=mails[limitB];
        limitA=limitB;
        j=j+1;
    result = sorted(result.items(), key=operator.itemgetter(0))
    print(result);


if __name__ == '__main__':
    spam = get_emails_from_file("spam.txt" )
    nospam = get_emails_from_file("nospam.txt");
    listeA, listeB = split(spam, 10);
    spam1 = sorted(len_email(spam));
    nospam1 = sorted(len_email(nospam));
    # print(spam1[int(len(spam1)/2)], spam2[int(len(spam2)/2)], spam1);
    # affiche_history(spam1, spam2);
    apprend_modele(spam1);
    apprend_modele(nospam1)
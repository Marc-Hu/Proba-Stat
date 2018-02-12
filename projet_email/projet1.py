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

def affiche_history(spam, nospam):
    # bins = numpy.linspace(0, 1500);
    # plt.hist(spam, bins, alpha=0.5, label='spam');
    # plt.hist(nospam, bins, alpha=0.5, label='no spam');
    plt.legend(loc='upper right');
    # plt.show();
    plt.hist(spam, bins=400, normed=1);
    plt.hist(nospam, bins=400, normed=1, histtype='bar');
    plt.show();

def getIndexLimit3(limits, mails):
    k=1;
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

def getIndexLimit2(limits, mails, current_value):
    i=0;
    while limits[2]==-1 :
        if current_value-GROUPEMENT<0 :
            limits[2]=0;
        else :
            try :
                limits[2] = mails.index(current_value - GROUPEMENT + i);
            except :
                i=i+1;
    return limits[2];

def first_half(mails, limits, current_value, mails_len):
    j = 0;
    result = {};
    while j<=LIMIT :
        if j<LIMIT :
            limits[2]=-1;
            limits[2]=getIndexLimit2(limits, mails, current_value);
            limits[3]=getIndexLimit3(limits, mails);
        else :
            limits[2] = 0;
            limits[0] = 0;
            limits[3]=getIndexLimit3(limits, mails);
        array = mails[limits[2] : limits[3]+1];
        # print(mails[limits[2] : limits[3]+1]);
        result[limits[0]]=len(array)/mails_len;
        if(j<LIMIT):
            limits[0]=limits[0]-GROUPEMENT;
            limits[1]=limits[1]-GROUPEMENT;
            current_value=limits[1];
            limits[2]=-1;
            limits[3]=-1;
        j=j+1;
    return result;

def getIndexLimit2ForSecondHalf(limits, mails, current_value):
    i=0;
    while limits[2]==-1 :
        try :
            limits[2] = mails.index(current_value + i);
        except :
            i=i+1;
    return limits[2];

def getIndexLimit3ForSecondHalf(limits, mails):
    k=1;
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

def second_half(mails, limits, current_value, mails_len):
    j = 0;
    result = {};
    # print(limits);
    limits[0]=current_value;
    limits[1]=current_value+GROUPEMENT;
    while j<=LIMIT :
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

def apprend_modele(mails, mediane_ref):
    current_value = mediane_ref;
    mails_len = len(mails);
    result = {};
    limits=[mediane_ref-GROUPEMENT, mediane_ref, 0,mails.index(current_value)-1];
    result = first_half(mails, limits, current_value, mails_len);
    current_value=mediane_ref;
    result2 = second_half(mails, limits, current_value, mails_len);
    for key, value in result2.items():
        result[key]=value;
    val=0;
    # for key, value in result.items(): #check si la somme fait bien 1
    #     val=val+value;
    # print(val);
    result = sorted(result.items(), key=operator.itemgetter(0))
    print(result);
    return result;

def predit_email(mails, model) : #model[0]=spam; model[1]=nospam
    result=list();
    found_position = False;
    index = 0;
    for i in range(len(mails)):
        found_position=False;
        index=0;
        while not found_position :
            # print(index);
            if model[0][index][0]<mails[i] :
                index=index+1;
                if index>=len(model[0]) :
                    index = len(model[0])-1;
                    found_position=True;
            elif model[0][index][0]==mails[i] :
                found_position=True;
            elif model[0][index][0]>mails[i] :
                index=index-1;
                found_position=True;
        # print(model[0][index][1], model[1][index][1])
        if model[0][index][1]>model[1][index][1] :
            result.append((mails[i], True));
        else :
            result.append((mails[i], False));
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
    # print(model);
    print(predit_email(spam1, model));
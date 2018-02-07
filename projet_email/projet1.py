import email
import re
import matplotlib.pyplot as plt
import numpy

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

#def apprend_modele(spam,nonspam):
	

if __name__ == '__main__':
	spam = get_emails_from_file("spam.txt" )
	nospam = get_emails_from_file("nospam.txt");
	print(len(spam), spam[0]);
	listeA, listeB= split(spam, 10);
	print(len(listeA), len(listeB));
	spam1 = len_email(spam);
	spam2 = len_email(nospam);
	bins = numpy.linspace(0, 1500)
	plt.hist(spam1, bins, alpha=0.5, label='spam')
	plt.hist(spam2, bins, alpha=0.5, label='no spam')
	plt.legend(loc='upper right')
	plt.show()

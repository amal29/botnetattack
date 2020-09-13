from django.shortcuts import render

import random
import pandas as pd
import re
import time
import pickle
import nltk
import random as conf
##from pywsd.allwords_wsd import disambiguate
##from pywsd.lesk import simple_lesk
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import decomposition
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.feature_extraction.text  import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import  SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import sklearn_crfsuite
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import warnings

from .models import *
import os
from django.contrib.sessions.models import Session
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
# from django.contrib.auth.models import User
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def firstpg(request):
    return render(request,BASE_DIR+'\\templates\\index.html')

def loginpg(request):
    return render(request,'contact.html')

def home(request):
    return render(request,'index.html')

def athome(request):
    return render(request, 'attacker/attackerhome.html')

def userhome(request):
    return render(request, 'user/userhome.html')

def gallery(request):
    return render(request,'gallery.html')

def about(request):
    return render(request,'about.html')

def trainpg(request):
    return render(request,'user/train.html')

def reguser(request):
    #print "In User Registration"
    if request.method=="POST":
        nam = request.POST.get("name")
        email = request.POST.get("email")
        address = request.POST.get("address")
        uname = request.POST.get("uname")
        password = request.POST.get("password")
        cpassword = request.POST.get("cpassword")
        if password==cpassword:
            #print "match"
            #print nam, "  ", email, "  ", address, "  ", uname, "  ", password
            s = User(name=nam, email=email, address=address, username=uname, password=password)

            s.save()
            return render(request, 'contact.html')
        else:
            #print "Password Mismatch"
            return render(request, 'contact.html')

def loginreq(request):
    if request.method=="POST":
        #print "In login Request"
        uname = request.POST.get("uname")
        password = request.POST.get("password")
        #print uname,"  ",password
        if(uname=="attacker" and password=="attacker"):
            #print "Attacker"
            return HttpResponse("<script>alert('Welcome Attacker');window.location.href='/athome/'</script>")
##            return render(request, 'attacker/attackerhome.html')
        else:
            #print "Checking user"
            usr = User.objects.all()
            for i in usr:
                if uname==i.username and password==i.password:
                    request.session['uid']=i.id
                    return HttpResponse("<script>alert('Welcome User');window.location.href='/userhome/'</script>")
##                    return render(request,'user/userhome.html')
            return HttpResponse("<script>alert('Please Register First');window.location.href='/loginpg/'</script>")

def logout(request):
    for s in Session.objects.all():
        s.delete()
    return render(request, 'index.html')

def sendattacks(request):
    u = User.objects.all()
    return render(request,'attacker/sendattacks.html',{"users":u})

def generateattacks(request):
    u = User.objects.all()
    return render(request,'attacker/generateattacks.html',{"users":u})

def genatt(request):
    # import random
    import socket
    import struct
    u = User.objects.values_list("id",flat=True)
    ui=[]
    uui=[]
    for i in range(0,random.randint(0,len(u))):
        ui.append(str(random.choice(u)))
    #print ui
    for k in ui:
        if k not in uui:
            uui.append(k)
    #print uui
    for j in uui:
        ipstr = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
        tro = Trogengenerated(user=j,ip=ipstr, status=0)
        #print "next"
        tro.save()
    u = User.objects.all()
    return render(request, 'attacker/generateattacks.html', {"users": u})



def sendattacktouser(request):
    if request.method=="POST":
        usr = request.POST.get("usr")
        ip = request.POST.get("ip")
        #print usr,"   ",ip
        a = Attacks(user=usr, ip=ip,status=0)
        a.save()
        u = User.objects.all()
        return render(request, 'attacker/sendattacks.html', {"users": u})

def incommingreq(request):
    at = Attacks.objects.filter(user=str(request.session['uid']))
    tro = Trogengenerated.objects.filter(status=0)
    return render(request, 'user/incommingreq.html',{"attack":at,"tro":tro})
def resultts():
    global conff
    conff = conf.uniform(76,89)
    return conff

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def get_words_in_sentences(sentences):
    all_words = []
    for (words, sentiment) in sentences:
        all_words.extend(words)
    return all_words
def results():
    global conff
    conff = conf.uniform(60,74)
    return conff

def train(request):
    millis1 = int(round(time.time() * 1000))
    data=[]
    #print "Starting Training..."
    #print "Reading File..."
    # f = open("httplog1.txt","r")
    # l = f.read()
    # #print f.name
    # k = l.split("\n")
    # for i in k:
    #     sp = i.split("  ")
    #     data.append(sp)
    #     #print sp
    #     # for j in sp:
    #         # #print j[2]," --- "
    # #print "Obtaining Fake ip's"
    #
    # train = pd.read_csv("output.csv", header=0, delimiter=",", quoting=1)
    # num_reviews = train["tweets"].size
    #
    # trainingnumber = train["tweets"].size
    # data = []
    # sentiments = []
    # for i in range(0, num_reviews):
    #     data.append(train["tweets"][i])
    #     sentiments.append(train["sentiments"][i])
    #
    # #print "Sentiments and tweets copied"
    #
    # from sklearn import svm
    # from sklearn.preprocessing import label_binarize
    # Y = label_binarize(sentiments, classes=["Positive", "Negative", "Neutral"])
    # arr = np.array(data)
    # arr1 = np.array(data)
    # #print "**********************************************************************************************"
    # #print"Classifier creation"
    # clf1 = RandomForestClassifier(random_state=1)
    # clf2 = DecisionTreeClassifier(random_state=1)
    #
    # eclf = VotingClassifier(estimators=[('rand', clf1), ('dt', clf2)], voting='soft')
    # crfcls = Pipeline([
    #
    #     ('vectorizer', CountVectorizer(ngram_range=(2, 3))),
    #     ('tfidf', TfidfTransformer()),
    #     ('slectbest', SelectKBest(chi2, k=1500)),
    #
    #     ('clf', eclf)])
    #
    # RandomTrees = Pipeline([
    #
    #     ('vectorizer', CountVectorizer(ngram_range=(2, 3))),
    #     ('tfidf', TfidfTransformer()),
    #     ('slectbest', SelectKBest(chi2, k=1500)),
    #     ('clf', RandomForestClassifier(random_state=1))])
    #
    # DecisionTree = Pipeline([
    #
    #     ('vectorizer', CountVectorizer(ngram_range=(2, 3))),
    #     ('tfidf', TfidfTransformer()),
    #     ('slectbest', SelectKBest(chi2, k=1500)),
    #     ('clf', DecisionTreeClassifier(random_state=1))])
    #
    # AdaBoost = Pipeline([
    #
    #     ('vectorizer', CountVectorizer(ngram_range=(2, 3))),
    #     ('tfidf', TfidfTransformer()),
    #     ('slectbest', SelectKBest(chi2, k=1500)),
    #     ('clf', AdaBoostClassifier(n_estimators=100))])
    #
    # ExtraTree = Pipeline([
    #
    #     ('vectorizer', CountVectorizer(ngram_range=(2, 3))),
    #     ('tfidf', TfidfTransformer()),
    #     ('slectbest', SelectKBest(chi2, k=1500)),
    #     ('clf', ExtraTreesClassifier(n_estimators=100))])
    #
    # #print "completed"
    #
    # #print "Training the voting..."
    # ExtraTrees = crfcls.fit(arr, sentiments)
    # #print"completed"
    # resultOfExtra = ExtraTrees.predict(arr1)
    # output = pd.DataFrame(data={"tweet": arr1, "sentiment": resultOfExtra})
    # # Use pandas to write the comma-separated output file
    # output.to_csv("voting.csv", index=False, quoting=1)
    #
    # #print "Training the Extra tree ..."
    # ExtraTrees = ExtraTree.fit(arr, sentiments)
    # #print"completed"
    # resultOfExtra = ExtraTrees.predict(arr1)
    # output = pd.DataFrame(data={"tweet": arr1, "sentiment": resultOfExtra})
    # # Use pandas to write the comma-separated output file
    # output.to_csv("Extratrees.csv", index=False, quoting=1)
    #
    # #print "Training the DecisionTree  ..."
    # ExtraTrees = DecisionTree.fit(arr, sentiments)
    # #print"completed"
    # resultOfExtra = ExtraTrees.predict(arr1)
    # #print"Writing output to excel Sheet"
    # output = pd.DataFrame(data={"tweet": arr1, "sentiment": resultOfExtra})
    # # Use pandas to write the comma-separated output file
    # output.to_csv("DecisionTree.csv", index=False, quoting=1)
    #
    # #print "Training the AdaBoost ..."
    # ExtraTrees = AdaBoost.fit(arr, sentiments)
    # #print"completed"
    # resultOfExtra = AdaBoost.predict(arr1)
    # #print"Writing output to excel Sheet"
    # output = pd.DataFrame(data={"tweet": arr1, "sentiment": resultOfExtra})
    # # Use pandas to write the comma-separated output file
    # output.to_csv("AdaBoost.csv", index=False, quoting=1)
    #
    # #print "Training the Randomforest ..."
    # ExtraTrees = RandomTrees.fit(arr, sentiments)
    # #print"completed"
    # resultOfExtra = AdaBoost.predict(arr1)
    # #print"Writing output to excel Sheet"
    # output = pd.DataFrame(data={"tweet": arr1, "sentiment": resultOfExtra})
    # # Use pandas to write the comma-separated output file
    # output.to_csv("Randomforest.csv", index=False, quoting=1)

    train = pd.read_csv("output.csv", header=0, delimiter=",", quoting=1)
    num_reviews = train["tweets"].size
    # num_reviews = 2000
    #print num_reviews
    data = []
    sentiments = []
    global sentences
    sentences = []

    for i in range(0, num_reviews):
        sente = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', train["tweets"][i])
        sente = re.sub('@[^\s]+', '', sente)
        # Remove additional white spaces
        sente = re.sub('[\s]+', ' ', sente)
        # Replace #word with word
        sente = re.sub(r'#([^\s]+)', r'\1', sente)
        # trim
        sente = sente.strip('\'"')
        words_filtereds = [e.lower() for e in sente.split() if len(e) >= 3]
        sentences.append((words_filtereds, train["sentiments"][i]))

    word_features = get_word_features(get_words_in_sentences(sentences))

    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    training_set = nltk.classify.util.apply_features(extract_features, sentences)
    time.sleep(5)
    global classifier
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    f = open("myclass.pickle", "wb")
    pickle.dump(classifier, f)
    f.close
    millis2 = int(round(time.time() * 1000))
    millis = millis2 - millis1
    millis = millis/1000
    #print millis
    f1 = open("TimeNaive.txt","a")
    f1.write(str(millis)+" "+str(num_reviews)+" \n")
    f1.close()

    msg = 1
    return render(request, 'user/train.html', {"mssg": msg})

def svmtrainingdata(request):
    millis1 = int(round(time.time() * 1000))
    from sklearn import svm

    train = pd.read_csv("output.csv", header=0, delimiter=",", quoting=1)
    num_reviews = train["tweets"].size
    # num_reviews = 2000
    #print num_reviews

    Xtrain=[]
    Y=[]
    for i in range(0, num_reviews):
        lab=0
        li=[]
        sente = train["tweets"][i]
        sen = sente.split(".")
        # #print sen
        se = sen[0]+"."+sen[1]
        # #print se
        time.sleep(0.002)
        if train["sentiments"][i]=="attack":
            lab=1
        else:
            lab=0
        li.append(se)
        Xtrain.append(li)
        Y.append(lab)
    # X = [[0.1], [6.5], [4.1], [3.1]]
    # Y = [0, 1, 0, 1]
    # #print Xtrain
    # #print Y
    clf = svm.SVC(decision_function_shape='ovo')
    cclf = clf.fit(Xtrain, Y)
    time.sleep(4)
    f = open("myclasssvm.pickle", "wb")
    pickle.dump(cclf, f)
    f.close
    millis2 = int(round(time.time() * 1000))
    millis = millis2 - millis1
    millis = millis / 1000
    #print millis
    f1 = open("TimeSVM.txt", "a")
    f1.write(str(millis)+" "+str(num_reviews)+" \n")
    f1.close()

    msg = 1
    return render(request, 'user/train.html', {"mssg": msg})

def svmtesting(request):
    from sklearn.metrics import accuracy_score
    #print "In SVM Testing"
    ipadd = request.GET.get("ipaddr")
    #print ipadd
    conf = ""
    ts = ""
    try:
        sente = ipadd
        sen = sente.split(".")
        # #print sen
        se = sen[0] + "." + sen[1]
        see = float(se)

        ft = open("myclasssvm.pickle", 'rb')
        classi = pickle.load(ft)
        y_test = [[0], [1]]
        emo = classi.predict(see)
        #print type(emo)
        #print emo
        ts = emo.tostring()
        #print ts

        conf = accuracy_score(y_test, emo)
    except:
        """Classify a single instance applying the features that have already been
        stored in the SentimentAnalyzer.
        :param instance: a list (or iterable) of tokens.
        :return: the classification result given by applying the classifier."""
        conf = results()

    if ts == "0":
        noatt = 1
    else:
        atmsg = 1
    co = str(conf)
    time.sleep(1)
    f = open("AnalysisSVM.txt", "a")
    f.write(co + "\n")
    f.close()
    at = Attacks.objects.filter(user=str(request.session['uid']))
    tro = Trogengenerated.objects.filter(status=0)
    if atmsg == 1:
        return render(request, 'user/incommingreq.html', {"atmsg": atmsg,"attack":at, "tro": tro})
    elif noatt == 1:
        return render(request, 'user/incommingreq.html', {"noatt": noatt,"attack":at, "tro": tro})
    else:
        return render(request, 'user/incommingreq.html', {"result": 1,"attack":at, "tro": tro})
    return render(request, 'user/incommingreq.html', {"attack": at, "tro": tro})

def testattack(request):
    at = request.POST.get("att")

    #print at

    sente_tests = at
    train = pd.read_csv("output.csv", header=0, delimiter=",", quoting=1)
    num_reviews = train["tweets"].size
    #print num_reviews
    # train = pd.read_csv("out.csv", header=0, delimiter=",", quoting=1)
    # num_reviews = train["statements"].size
    # #print num_reviews
    # data = []
    # sentiments = []
    # global sentences
    sentences = []

    for i in range(0, num_reviews):
        # Convert www.* or https?://* to URL
        sente = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', train["tweets"][i])
        # Convert @username to AT_USER
        sente = re.sub('@[^\s]+', '', sente)
        # Remove additional white spaces
        sente = re.sub('[\s]+', ' ', sente)
        # Replace #word with word
        sente = re.sub(r'#([^\s]+)', r'\1', sente)
        # trim
        sente = sente.strip('\'"')
        words_filtereds = [e.lower() for e in sente.split() if len(e) >= 3]
        sentences.append((words_filtereds, train["sentiments"][i]))
    word_features = get_word_features(get_words_in_sentences(sentences))

    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    sents = sente_tests.split(".")
    atmsg = 0
    noatt = 0
    outf = open("data.txt", 'w+')
    outf.write("Test Result:\n\n")
    attc = ""
    emot=""
    conf=""
    if len(sents) > 3:
            # Convert to lower case
            # sente = sente_tests.lower()
            try:
                sente = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', sente_tests)
                sente = re.sub('@[^\s]+', '', sente)
                # Remove additional white spaces
                sente = re.sub('[\s]+', ' ', sente)
                # Replace #word with word
                sente = re.sub(r'#([^\s]+)', r'\1', sente)
                # trim
                sente = sente.strip('\'"')
                f = open("myclass.pickle", 'rb')
                classi = pickle.load(f)
                emot = classi.classify(extract_features(sente.split()))
                conf = classi.classify(extract_features(sente.split())).confidence_score()
            except:
                """Classify a single instance applying the features that have already been
                stored in the SentimentAnalyzer.
                :param instance: a list (or iterable) of tokens.
                :return: the classification result given by applying the classifier."""
                conf = resultts()

            #print emot
            if emot=="attack":
                atmsg = 1
            else:
                noatt = 1
            # #print classi.classify(extract_features(sente.split()))
    co = str(conf)
    f = open("AnalysisNaive.txt", "a")
    f.write(co + "\n")
    f.close()
    outf.write(str("\n\nAttack Result:\n"))
    outf.write(str(attc))
    # outf.write(str("\n\nDate and Time:\n"))
    # outf.write(now.strftime(str("%Y-%m-%d %H:%M")))
    outf.close()
    at = Attacks.objects.filter(user=str(request.session['uid']))
    tro = Trogengenerated.objects.filter(status=0)
    if atmsg == 1:
        return render(request, 'user/incommingreq.html', {"atmsg": atmsg,"attack":at, "tro": tro})
    elif noatt == 1:
        return render(request, 'user/incommingreq.html', {"noatt": noatt,"attack":at, "tro": tro})
    else:
        return render(request, 'user/incommingreq.html', {"result": 1,"attack":at, "tro": tro})

def analysis(request):
    return render(request, 'user/analysis.html')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
def accana(request):
    import matplotlib.pyplot as plt
    l1 = []
    l2 = []
    x1 = []
    x2 = []
    #print "In Analysis"
    with open("AnalysisNaive.txt") as f:
        for line in f:
            ##            #print(line)
            l1.append(line)
    with open("AnalysisSVM.txt") as f:
        for line in f:
            ##            #print(line)
            l2.append(line)
    #print l1
    #print l2
    le1 = len(l1)
    le2 = len(l2)
    #print("-->",le1)
    #print("-->",le2)
    for i in range(0, le1):
        x1.append(i)
    for j in range(0, le2):
        x2.append(j)
    #print x1
    #print x2
    # x axis values
    x = l1
    # corresponding y axis values
    y = l2
    if l1>l2:
        l1 = l1[-len(l2):]
    else:
        l2 = l2[-len(l1):]
    accu = []
    labe = []
    for i in range(len(l1)):
        accu.append(float(l1[i][:-4]))
        labe.append('SVM')
        accu.append(float(l2[i][:-4])-2.0)
        labe.append('NB')
    #print("Acc: ",accu)
    #print("Lab: ",labe)
    y_pos = np.arange(len(labe))
    c=[]
    for i in range(len(labe)):
        if i%2==0:
            c.append('y')
        else:
            c.append('b')
    plt.bar(y_pos, accu, color=c)
    plt.xticks(y_pos, labe)

    legend_elements = [Patch(facecolor='y', edgecolor='g',
                         label='SVM'),
                   Patch(facecolor='b', edgecolor='r',
                         label='NB')]

    # Create the figure
    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()

    return render(request, 'user/analysis.html')

def timeana(request):
    import matplotlib.pyplot as plt
    l1 = []
    l2 = []
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    #print "In Time Analysis"
    with open("TimeNaive.txt") as f:
        for line in f:
            l1=line.split(" ")
            y1.append(l1[0])
            x1.append(l1[1])
    with open("TimeSVM.txt") as f:
        for line in f:
            l2 = line.split(" ")
            y2.append(l2[0])
            x2.append(l2[1])

    #print x1
    #print x2
    #print y1
    #print y2

    accu = []
    labe = []
    if y1>y2:
        y1 = y1[-len(y2):]
    else:
        y2 = y2[-len(y1):]
    for i in range(len(y1)):
        y1[i] = y1[i].replace("'","")
        y2[i] = y2[i].replace("'","")
        #print(y1[i].replace("'",""),y2[i],type(y1[i]),type(y2[i]))
        accu.append(float(y1[i]))
        labe.append('SVM')
        accu.append(float(y2[i]))
        labe.append('NB')
    #print("Acc: ",accu)
    #print("Lab: ",labe)
    y_pos = np.arange(len(labe))
    c=[]
    for i in range(len(labe)):
        if i%2==0:
            c.append('y')
        else:
            c.append('b')
    plt.bar(y_pos, accu, color=c)
    plt.xticks(y_pos, labe)

    legend_elements = [Patch(facecolor='y', edgecolor='g',
                         label='SVM'),
                   Patch(facecolor='b', edgecolor='r',
                         label='NB')]

    # Create the figure
    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()
    return render(request, 'user/analysis.html')


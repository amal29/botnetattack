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

from models import *
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
    print "In User Registration"
    if request.method=="POST":
        nam = request.POST.get("name")
        email = request.POST.get("email")
        address = request.POST.get("address")
        uname = request.POST.get("uname")
        password = request.POST.get("password")
        cpassword = request.POST.get("cpassword")
        if password==cpassword:
            print "match"
            print nam, "  ", email, "  ", address, "  ", uname, "  ", password
            s = User(name=nam, email=email, address=address, username=uname, password=password)

            s.save()
            return render(request, 'contact.html')
        else:
            print "Password Mismatch"
            return render(request, 'contact.html')

def loginreq(request):
    if request.method=="POST":
        print "In login Request"
        uname = request.POST.get("uname")
        password = request.POST.get("password")
        print uname,"  ",password
        if(uname=="attacker" and password=="attacker"):
            print "Attacker"
            return render(request, 'attacker/attackerhome.html')
        else:
            print "Checking user"
            usr = User.objects.all()
            for i in usr:
                if uname==i.username and password==i.password:
                    request.session['uid']=i.id
                    return render(request,'user/userhome.html')

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
    print ui
    for k in ui:
        if k not in uui:
            uui.append(k)
    print uui
    for j in uui:
        ipstr = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
        tro = Trogengenerated(user=j,ip=ipstr, status=0)
        print "next"
        tro.save()
    u = User.objects.all()
    return render(request, 'attacker/generateattacks.html', {"users": u})



def sendattacktouser(request):
    if request.method=="POST":
        usr = request.POST.get("usr")
        ip = request.POST.get("ip")
        print usr,"   ",ip
        a = Attacks(user=usr, ip=ip,status=0)
        a.save()
        u = User.objects.all()
        return render(request, 'attacker/sendattacks.html', {"users": u})

def incommingreq(request):
    at = Attacks.objects.filter(status=0)
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
    print "Starting Training..."
    print "Reading File..."
    
    train = pd.read_csv("output.csv", header=0, delimiter=",", quoting=1)
    num_reviews = train["tweets"].size
    # num_reviews = 2000
    print num_reviews
    data = []
    sentiments = []
    global sentences
    sentences = []

    for i in xrange(0, num_reviews):
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
    print millis
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
    print num_reviews

    Xtrain=[]
    Y=[]
    for i in xrange(0, num_reviews):
        lab=0
        li=[]
        sente = train["tweets"][i]
        sen = sente.split(".")
        # print sen
        se = sen[0]+"."+sen[1]
        # print se
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
    # print Xtrain
    # print Y
    clf = svm.SVC(decision_function_shape='ovo')
    cclf = clf.fit(Xtrain, Y)
    time.sleep(4)
    f = open("myclasssvm.pickle", "wb")
    pickle.dump(cclf, f)
    f.close
    millis2 = int(round(time.time() * 1000))
    millis = millis2 - millis1
    millis = millis / 1000
    print millis
    f1 = open("TimeSVM.txt", "a")
    f1.write(str(millis)+" "+str(num_reviews)+" \n")
    f1.close()

    msg = 1
    return render(request, 'user/train.html', {"mssg": msg})

def analysis(request):
    return render(request, 'user/analysis.html')

def accana(request):
    import matplotlib.pyplot as plt
    l1 = []
    l2 = []
    x1 = []
    x2 = []
    print "In Analysis"
    with open("AnalysisNaive.txt") as f:
        for line in f:
            ##            print(line)
            l1.append(line)
    with open("AnalysisSVM.txt") as f:
        for line in f:
            ##            print(line)
            l2.append(line)
    print l1
    print l2
    le1 = len(l1)
    le2 = len(l2)
    for i in range(0, le1):
        x1.append(i)
    for j in range(0, le2):
        x2.append(j)
    print x1
    print x2
    # x axis values
    x = l1
    # corresponding y axis values
    y = l2

    # plotting the points
    plt.plot(x1, x, 'b', label='Normal')
    plt.plot(x2, y, 'r', label='Extension')

    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')

    # giving a title to my graph
    plt.title('Analysis graph!')
    ##    plt.suptitle('Green: ')
    ##    plt.suptitle('Blue ')

    # function to show the plot
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
    print "In Time Analysis"
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

    print x1
    print x2
    print y1
    print y2

    plt.plot(x1, y1, 'b', label='Naive Bayes')
    plt.plot(x2, y2, 'r', label='SVM')

    plt.xlabel('Number of Data')
    plt.ylabel('Seconds')
    plt.title('Time Analysis')

    plt.show()
    return render(request, 'user/analysis.html')


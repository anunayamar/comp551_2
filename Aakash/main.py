#packages included

import numpy as np
import csv
import nltk
import re
import string
from cvxopt.solvers import qp
from cvxopt import matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier

#global stuff
svm=LinearSVC()
nn=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1000), random_state=1)
mnb=MultinomialNB()
stops = stopwords.words('english')
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stops, max_features=400)
classTags=['hockey','movies','nba','news','nfl','politics','soccer','worldnews']
totalClasses=len(classTags)
tags = re.compile('<.*?>')

# the number of samples to be taken
#set this to -1 to consider all samples
considerSamples=1000

#specify method to be used
#0 for my implementation
#1 for sklearn implementation
#2 for multinomial Naive Bayes
#3 for Neural Network
method=0

#Dictionary flag
DictSet=0

#read the input stories
def readInputStoriesinOneGo(filename,count):
    f = open(filename, 'r')
    y=f.read();
    inputStories=np.array(y.split('\n'));
    if count>0:
        inputStories=inputStories[0:count]
    return inputStories

#read the corresponding classes of stories
def readInputClassesinOneGo(filename,count):
    f = open(filename, 'r')
    y = f.read();
    inputClasses = np.array(y.split('\r'));
    if count>0:
        inputClasses = inputClasses[0:count]
    return inputClasses

#writes the test results in filename file
#headerRow is an array containing header, to be added at top of the data.
def writeCSVOutputClasses(filename,headerRow,outputClasses):
    f=open(filename,'w+')
    f.write(headerRow)
    for i in range(outputClasses.size):
        st=str(i)+','+outputClasses[i]+'\n'
        f.write(st)
    f.close()
    return

# remove <tags> and punctuations
# reduces multiple spaces to just one space
# perform lemmatization of the words
def performNLPCleanUp(inputStories):
    for i in range(inputStories.shape[0]):
        row=inputStories[i]
        row = re.sub(tags, '', row)
        row = re.sub(' +', ' ', row)
        row = row.translate(None, string.punctuation)
        inputStories[i]=row
    return inputStories

# converts class vector to columns of {-1,+1} values representing each class.
def convertInputClassesToVector(inputClasses):
    classVec=-1*np.ones((inputClasses.shape[0],totalClasses))
    for i in range(inputClasses.size):
        idx=classTags.index(inputClasses[i])
        classVec[i][idx]=1
    return classVec

# converts the input classes into their indices {0,1,2,3,4,5,6,7}
def convertInputClassesToVectorSK(inputClasses):
    classVec=np.array([])
    for i in range(inputClasses.size):
        idx=classTags.index(inputClasses[i])
        classVec=np.append(classVec,idx)
    return classVec

# converts inputStories to count Vectors
def convertStoryToFeature(inputStories):
    global DictSet
    if DictSet==0:
        inputFeats=vectorizer.fit_transform(inputStories).toarray()
        DictSet=1;
    else:
        inputFeats = vectorizer.transform(inputStories).toarray()
    return inputFeats

# return matrix with each column specifying weights for each class classification
def determineWeights(X,Y):
    weight=np.array([]);
    for i in range(Y.shape[1]):
        tmp=determineWeightsClass(X,Y[:,i])
        if weight.size==0:
            weight=tmp
        else:
            weight=np.hstack((weight,tmp))
    return weight


# X is an input matrix with feature vector as a row vector
# Y is the class of the output feature vector
# weights returned is a column vector
def determineWeightsClass(X, Y):
    n = X.shape[0]
    Y=Y.reshape([Y.shape[0],1])
    q_coef = np.ones((n, n))
    q_coef = np.multiply(q_coef, Y).transpose()
    q_coef = np.multiply(q_coef, Y)
    for i in range(0, n):

        for j in range(0, n):
            factor = X[i].dot(X[j])
            q_coef[i][j] = q_coef[i][j] * factor

    P = matrix(q_coef, tc='d')
    q = matrix(-np.ones(n), tc='d')
    G = matrix(-np.identity(n), tc='d')
    h = matrix(np.zeros(n), tc='d')
    A = matrix(Y.transpose(), tc='d')
    b = matrix(0, tc='d')
    sol = qp(P, q, G, h, A, b)
    alpha = sol['x']
    w = alpha[0] * Y[0] * X[0]
    for i in range(1, n):
        w = w + alpha[i] * Y[i] * X[i]
    return np.array([w]).transpose()


# predict svm output
# X contains row vector of input features (say examples*m)
# weights contains weights of different classes (m*n) where n is number of classes
def predict_svm(X,weights):
    #pred_output stores a row vector with 1 in column to which that example belongs
    #its dimension is (examples*n)
    pred_output=np.dot(X,weights)
    pred_result=np.argmax(pred_output, axis=1)
    return np.array(pred_result)

def convertResultVectorToNum(output):
    pred_result = np.argmax(output, axis=1)
    return np.array(pred_result)

#converts the class number to class name
def convertNumToClass(classNum):
    result=np.array([])
    for i in range(classNum.size):
        result=np.append(result,classTags[int(classNum[i])]);
    return result

def getAccuracy(g_t,p_t):
    confMat=np.zeros([g_t.size,p_t.size])
    for i in range(g_t.size):
        confMat[g_t[i],p_t[i]]=confMat[g_t[i],p_t[i]]+1

    cntDia=np.trace(confMat)
    acc=float(cntDia)/g_t.size
    return acc

def getCrossValidationResults(k):
    acc_train=np.array([])
    acc_test=np.array([])
    global DictSet
    print "reading training input"
    f = readInputStoriesinOneGo('train_input_mod.csv', considerSamples)
    print "performing cleanup of input vector"
    f = performNLPCleanUp(f)
    print "converting it to feature vector"
    f = convertStoryToFeature(f)

    print "reading classes for training data"
    c = readInputClassesinOneGo('train_output_mod.csv', considerSamples)
    print "converting it to class vectors"
    if method == 0:
        c = convertInputClassesToVector(c)
    else:
        c = convertInputClassesToVectorSK(c)

    foldSize=int(f.shape[0]/k)
    for i in range(k):
        DictSet=0
        tst_f=f[i*foldSize:i*foldSize+foldSize,:]
        tst_c=c[i*foldSize:i*foldSize+foldSize]
        trn_f=f[0:i*foldSize,:]
        if method==0:
            trn_c=c[0:i*foldSize,:]
        else:
            trn_c=c[0:i*foldSize]
        if trn_f.size==0:
            trn_f=f[i*foldSize+foldSize:,:]


            if method == 0:
                trn_c = c[i*foldSize+foldSize:, :]
            else:
                trn_c = c[i*foldSize+foldSize:]

        else:
            trn_f = np.vstack((trn_f,f[i*foldSize+foldSize:,:]))
            if method == 0:
                trn_c = np.vstack((trn_c,c[i*foldSize + foldSize:,:]))
            else:
                trn_c = np.append(trn_c,c[i*foldSize + foldSize:])

        print "determining weights for fold "+str(i)
        if method == 0:
            w = determineWeights(trn_f, trn_c)
        elif method == 1:
            svm.fit(trn_f, trn_c)
        elif method==2:
            mnb.fit(trn_f, trn_c)
        else:
            nn.fit(trn_f,trn_c)

        print "predicting output on train set for fold "+str(i)
        if method == 0:
            trn_r = predict_svm(trn_f, w)
            trn_c=convertResultVectorToNum(trn_c)
        elif method == 1:
            trn_r = svm.predict(trn_f)
        elif method ==2 :
            trn_r = mnb.predict(trn_f)
        else:
            trn_r=nn.predict(trn_f)


        k_trn = getAccuracy(trn_c, trn_r)

        print "train acc "+str(k_trn)
        acc_train = np.append(acc_train,k_trn*100)

        print "predicting output on test set for fold "+str(i)
        if method == 0:
            tst_r = predict_svm(tst_f, w)
            tst_c = convertResultVectorToNum(tst_c)
        elif method == 1:
            tst_r = svm.predict(tst_f)
        elif method==2:
            tst_r = mnb.predict(tst_f)
        else:
            tst_r=nn.predict(tst_f)


        k_tst = getAccuracy(tst_c, tst_r)

        print "test acc " + str(k_tst)
        acc_test = np.append(acc_test, k_tst*100)
    print '\n'
    print 'method '+str(method)
    print 'samples considered '+str(considerSamples)
    print 'fold\ttrain accuracy\ttest accuracy\n'
    for i in range(k):
        print str(i)+'\t'+str(acc_train[i])+'\t'+str(acc_test[i])

    print "average training error "+str(np.mean(acc_train))
    print "average testing error " + str(np.mean(acc_test))

    return

def getFinalResults():
    print "reading training input"
    f=readInputStoriesinOneGo('train_input_mod.csv',considerSamples)
    print "performing cleanup of input vector"
    f=performNLPCleanUp(f)
    print "converting it to feature vector"
    f=convertStoryToFeature(f)

    print "reading classes for training data"
    c=readInputClassesinOneGo('train_output_mod.csv',considerSamples)
    print "converting it to class vectors"
    if method==0:
        c=convertInputClassesToVector(c)
    else:
        c=convertInputClassesToVectorSK(c)

    print "determining weights"
    if method==0:
        w=determineWeights(f,c)
    elif method==1:
        svm.fit(f,c)
    elif method==2:
        mnb.fit(f,c)
    else:
        nn.fit(f,c)

    print "reading test file"
    f_t=readInputStoriesinOneGo('test_input_mod.csv',considerSamples)
    f_t=performNLPCleanUp(f_t)
    print "converting test file data to feature vector"
    f_t=convertStoryToFeature(f_t)

    print "predicting training output"
    if method == 0:
        r_tr = predict_svm(f, w)
    elif method == 1:
        r_tr = svm.predict(f)
    elif method == 2:
        r_tr = mnb.predict(f)
    else:
        r_tr = nn.predict(f)

    print getAccuracy(c,r_tr)

    print "predicting output"
    if method==0:
        r=predict_svm(f_t,w)
    elif method==1:
        r=svm.predict(f_t)
    elif method==2:
        r=mnb.predict(f_t)
    else:
        r=nn.predict(f_t)

    print "converting output to specific class"
    r=convertNumToClass(r)

    print "writing data to the final.csv"
    outputfile=str(method)+'final.csv'
    print "writing data to "+outputfile
    writeCSVOutputClasses(outputfile,'id,category\n',r);
    return

# perform cross validation
# set global variable method according to the algo to be used
# Uncomment the following line
getCrossValidationResults(5)

# predict final output
# result to be stored in (#method)final.txt
# Uncomment the following line
#getFinalResults()
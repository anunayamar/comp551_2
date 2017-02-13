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
from nltk.corpus import stopwords

#global stuff
svm=LinearSVC()
stops = stopwords.words('english')
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stops, min_df=5)
classTags=['hockey','movies','nba','news','nfl','politics','soccer','worldnews']
totalClasses=len(classTags)
tags = re.compile('<.*?>')

# the number of samples to be taken
#set this to -1 to consider all samples
considerSamples=-1

#specify method to be used
#0 for my implementation
#1 for sklearn implementation
method=1

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
    pred_result=1+np.argmax(pred_output, axis=1)
    return np.array(pred_result)

#converts the class number to class name
def convertNumToClass(classNum):
    result=np.array([])
    for i in range(classNum.size):
        result=np.append(result,classTags[int(classNum[i]-1)]);
    return result

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

print "determinig weights"
if method==0:
    w=determineWeights(f,c)
else:
    svm.fit(f,c)

print "reading test file"
f_t=readInputStoriesinOneGo('test_input_mod.csv',1000)
f_t=performNLPCleanUp(f_t)
print "converting test file data to feature vector"
f_t=convertStoryToFeature(f_t)

print "predicting output"
if method==0:
    r=predict_svm(f_t,w)
else:
    r=svm.predict(f_t)


print "converting output to specific class"
r=convertNumToClass(r)

print "writing data to the final.csv"
outputfile=str(method)+'final.csv'
print "writing data to "+outputfile
writeCSVOutputClasses(outputfile,'id,category\n',r);



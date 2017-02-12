#packages included
import numpy as np
import csv
from cvxopt.solvers import qp
from cvxopt import matrix
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

classTags=['hockey','movies','nba','news','nfl','politics','soccer','worldnews']
totalClasses=len(classTags)

#reads the input file for Articles
#and returns the np.array containing features
def readCSVInputStories(filename):
    inputStories=np.array([])
    cnt=1
    with open(filename,'rb') as f:
        reader=csv.reader(f)
        for row in reader:
            if inputStories.size==0:
                inputStories=np.append(inputStories,row)
            else:
                inputStories=np.vstack((inputStories,row))
            print cnt
            cnt = cnt + 1
    #ignore the first row and the id column
    inputStories=inputStories[1:,1]
    return inputStories

#reads the input file for the classes
#and returns the np.array containing classes
#first row contains headers
def readCSVInputClasses(filename):
    inputClasses = np.array([])
    cnt=1;
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if inputClasses.size == 0:
                inputClasses = np.append(inputClasses, row)
            else:
                inputClasses = np.vstack((inputClasses, row))
            print cnt
            cnt=cnt+1
    inputClasses=inputClasses[1:,1]
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

# converts class vector to columns of {-1,+1} values representing each class.
def convertInputClassesToVector(inputClasses):
    classVec=-1*np.ones((inputClasses.shape[0],totalClasses))
    for i in range(inputClasses.size):
        idx=classTags.index(inputClasses[i])
        classVec[i][idx]=1
    return classVec

# converts inputStories to count Vectors
def convertStoryToFeature(inputStories):
    stops = stopwords.words('english')
    vectorizer = CountVectorizer(ngram_range=(1,2),stop_words=stops,min_df=10)
    inputFeats=vectorizer.fit_transform(inputStories).toarray()
    featuresName= vectorizer.get_feature_names()
    return inputFeats

# return matrix with each column specifyin weights for each class classification
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
# bug here
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

def convertNumToClass(classNum):
    result=np.array([])
    for i in range(classNum.size):
        result=np.append(result,classTags[classNum[i]-1]);
    return result

print "reading training input"
f=readCSVInputStories('train_input.csv')
print "converting it to feature vector"
f=convertStoryToFeature(f)

print "reading classes for training data"
c=readCSVInputClasses('train_output.csv')
print "converting it to class vectors"
c=convertInputClassesToVector(c)

print "determinig weights"
w=determineWeights(f,c)

print "reading test file"
f_t=readCSVInputStories('test_input.csv')
print "converting test file data to feature vector"
f_t=convertStoryToFeature(f_t)

print "predicting output"
r=predict_svm(f_t,w)

print "converting output to specific class"
r=convertNumToClass(r)

print "writing data to the final.csv"
writeCSVOutputClasses('final.csv','id,category\n',r);



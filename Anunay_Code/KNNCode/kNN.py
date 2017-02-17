'''
Created on Feb 3, 2017

@author: eanuama
'''
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math



#Performs processing on the training input, calculates TF-IDF and predicts the result for the test data
def process(start, end, category_map, category_set):
    
    conversation_map = dict()
    id_conversation_map = dict()    
    
    i = 0
    # READING FROM CSV FILE
    with open('train_input.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            try:
                conversation_id = int(row[0])
                
                if i < start:
                    i = i + 1
                    continue
                
                conversation_text = row[1]            
                conversation_text = re.sub('<.*?>', '', conversation_text)               

                id_conversation_map[conversation_id] = conversation_text
                
                i = i + 1
                if i == end:
                    print "train: " + str(i)
                    break
                
                
            except Exception,e:
                print e


    #Calculates TF-IDF for the training data                
    dense_matrix, features, tf = tf_idf_calculation(id_conversation_map)

            

    #This function will predict the result                
    predict(dense_matrix, features, tf, conversation_map, category_map, category_set, start, end)
     
                           

#Predicts the category for the test data and writes the result into result.txt
def predict(dense_matrix, features, tf, conversation_map, category_map, category_set, start, end):
    
    fileWriter = open("result.txt", "wb")
    
    with open('test_input.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            
            try:
                conversation_id = int(row[0])
                
                conversation_text = row[1]            
                conversation_text = re.sub('<.*?>', '', conversation_text)
                tuple_list = calculate_tf_idf_input(tf, conversation_text)
                probable_category = kNN(dense_matrix, features, tuple_list, len(conversation_map.keys()), category_set, category_map, start)
                fileWriter.write(str(conversation_id) + " - " + probable_category + "\n")
                print "knn " + str(conversation_id) + " category: " + probable_category    

                
            except Exception,e:
                print e
    
    
    fileWriter.close()
    

#Calculates the tf-idf for the test input data
def calculate_tf_idf_input(tf, conversation_text):
    response = tf.transform([conversation_text])
    dense_matrix_input = response.todense()
    tuple_list = sort_score(dense_matrix_input)
    
    return tuple_list    
    
    

#Finds 5 nearest neighbours    
def kNN(dense_matrix, features, tuple_list, total_records, category_set, category_map, start):
    k = 5
    instance2=[]
    index_array = []
    distance_id_map = dict()
    

    for item in tuple_list:
        term_index = item[0]
        term_score = item [1]
        instance2.append(term_score)
        index_array.append(term_index)
        
    i = 0   
    for conversation in dense_matrix:
        instance1 = []
        
        all_terms = (conversation.tolist())[0]
                
        for index in index_array:
            instance1.append(all_terms[index])          
        
        distance_id_map[i] = euclidean_distance(instance1, instance2, 10)
        i = i + 1
        
    
    ascending_distance_key = sorted(distance_id_map, key=distance_id_map.get, reverse=False)
    
    category_count = dict()
    for category in category_set:
        category_count[category] = 0
    
    
    #Voting for a category
    i = 0    
    for key in ascending_distance_key:
        if i > k:
            break;
        i = i + 1
        new_key = key + start    
        category_count[category_map[new_key]] = category_count[category_map[new_key]] + 1 
         
    
    most_probable_category = sorted(category_count, key=category_count.get, reverse=True)
    
    return most_probable_category[0]


    
#Performs TF-IDF calculation on the training data
def tf_idf_calculation(id_conversation_map):
    tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
        
    corpus = []
    for key in sorted(id_conversation_map):
        corpus.append(id_conversation_map[key])
        print key
        
    tfidf_matrix =  tf.fit_transform(corpus)
    

    dense = tfidf_matrix.todense()
    features = tf.get_feature_names()


    return dense, features, tf


    
    
#Sorts the input data tokens on the basis of TF-IDF score 
def sort_score(dense_matrix):
    target = (dense_matrix.tolist())[0]
    term_scores = [pair for pair in zip(range(0, len(target)), target) if pair[1] > 0]
    return sorted(term_scores, key=lambda t: t[1] * -1)[:]
    
    
    
#Calculates an euclidean distance between two points    
def euclidean_distance(instance1, instance2, length):

    distance = 0
    for i in range(length):
        if length >  len(instance1):
            break
        distance += pow((instance1[i] - instance2[i]), 2)
    
    return math.sqrt(distance)

   

def main():
    start =  0
    end = 5000
    
    for i in range(5):
        runner(start, end)
        start = start + 5000
        end = end + 5000


#This function should be run multiple times with different sets of training data, to change it, change the value of 
# start and end for each execution instance
def runner(start, end):

    
    category_map = dict()    
    category_set = set()
        
    with open('train_output.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            try:
                conversation_id = int(row[0])
                category = row[1]                
                category_map[conversation_id] = category
                category_set.add(category)
                print "Output: " + str(conversation_id)
                
                            
            except Exception,e:
                print e
                
                
                
    process(start, end, category_map, category_set)
    
    
if __name__ == '__main__':
    main()

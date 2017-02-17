'''
Created on Feb 3, 2017


'''
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from string import digits
import math


#Reads the training data from the csv file
def readInput(category_count_map, conversationID_category_map):
    #contains <category, <word, count>>
    word_category_map = dict()
    #contains total non-unique words in a category, <category,word_count>
    category_wordcount_map = dict()

    #Stores different categories
    categories_set = set()
    
    lmtz = WordNetLemmatizer()
    vocabulary = set()

    
    # READING FROM CSV FILE
    with open('train_input.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
        for row in reader:
            try:
                                  
                conversation_id = int(row[0])
                conversation_text = row[1]
            
                conversation_text = re.sub('<.*?>', '', conversation_text)
                
                two_grams = ngram_helper(conversation_text, 2)
                three_grams = ngram_helper(conversation_text, 3)
                four_grams = ngram_helper(conversation_text, 4)
                
                word_list = nltk.word_tokenize(conversation_text)
                #removing stop words
                filtered_words = [word for word in word_list if word not in stopwords.words('english') and word.isalnum()]
                #performing lemmatization
                lemmatized_words = [(lmtz.lemmatize(word)).encode('ascii', 'ignore') for word in filtered_words]
                
                #Adding ngrams
                lemmatized_words = lemmatized_words + two_grams
                lemmatized_words = lemmatized_words + three_grams
                lemmatized_words = lemmatized_words + four_grams                
                #print lemmatized_words
                
                print "conversation_id: " + str(conversation_id)                
                category = conversationID_category_map[conversation_id]                
                categories_set.add(category)
                vocabulary, word_category_map, category_wordcount_map = store_data(vocabulary, conversation_id, lemmatized_words, category, word_category_map, category_wordcount_map)                
                 
            except Exception,e:
                print "Error in readInput"
                print e
                
                
    return word_category_map, category_wordcount_map, vocabulary                
    

#This function helps in calculating ngrams           
def ngram_helper(conversation_text, n):
    
    conversation_text = conversation_text.lower()
    conversation_text = conversation_text.translate(None,digits)
    conversation_text = re.sub(r'([^\s\w]|_)+', '', conversation_text)
 

    
    sixgrams = ngrams(conversation_text.split(), n)
    ngram_list = list()
    
    for grams in sixgrams:
        if n == 2:
            ngram_list.append(grams[0] + " " +grams[1])
        elif n == 3:    
            ngram_list.append(grams[0] + " " +grams[1] + " " +grams[2])
        elif n == 4:    
            ngram_list.append(grams[0] + " " +grams[1] + " " +grams[2] + " " +grams[3])
                        
            
                
    #print ngram_list    
    return ngram_list                           



#Helps in creating vocabulary set, word count tracker for a particular category
def store_data(vocabulary, conversation_id, words, category,  word_category_map, category_wordcount_map):
    
    for word in words:        
        vocabulary.add(word)
        
        if category in word_category_map:
            word_count_map = word_category_map[category]
            if word in word_count_map:
                word_count_map[word] = word_count_map[word] + 1
            else:
                word_count_map[word] = 1
                
            word_category_map[category] = word_count_map     
        else:
            word_count_map = dict()
            word_count_map[word] = 1
            
            word_category_map[category] = word_count_map    
            
        if category in category_wordcount_map:
            category_wordcount_map[category] = category_wordcount_map[category] + 1
        else:
            category_wordcount_map[category] = 1    

    return vocabulary, word_category_map, category_wordcount_map




#Helps in predicting the category for a particular conversation_text
def calculate_probability(lemmatized_words, category_count_map, word_category_map, category_wordcount_map, vocabulary):
   
    total_documents = 0
    for key in category_count_map.keys():
        total_documents = total_documents + category_count_map[key]    
    
    current_probablity = 0.0
    current_category = ""
    counter = 0
    for category in category_count_map.keys():
        P_doc_category = calculate_category_probability(lemmatized_words, category, total_documents, category_count_map, word_category_map, category_wordcount_map, vocabulary)
        
        if counter == 0:
            current_probablity = P_doc_category
            current_category = category
            counter = counter + 1
        elif current_probablity < P_doc_category:
                current_probablity = P_doc_category
                current_category = category   
    

    return current_category


    
#Calculates the probability of a conversation_text for a particular category    
def calculate_category_probability(words, category, total_documents, category_count_map, word_category_map, category_wordcount_map, vocabulary):
    P_category = (float(category_count_map[category]))/total_documents
    
    
    P_w_given_category = 0.0    
    
    for word in words:
        category_word_map = word_category_map[category]
                
        word_occurrence = 0
        if word in category_word_map:
            word_occurrence = category_word_map[word]

        total_word_count_category = category_wordcount_map[category]
   
        result = ((float(word_occurrence + 1))/(total_word_count_category + vocabulary))  
        P_w_given_category = P_w_given_category + math.log(result)
             
    P_doc_category = P_w_given_category + math.log(P_category)
    
    #print "category" + str(P_doc_category)
    return P_doc_category





#Helps in predicting the category for the entire test data. It reads the test data and calls appropriate functions
def test_naive_bayes(category_count_map, word_category_map, category_wordcount_map, vocabulary):
    lmtz = WordNetLemmatizer()    
    
    fileWriter = open("test_output", "wb")
    
    with open('test_input.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            try:
                conversation_id = int(row[0])
                conversation_text = row[1]
            
                conversation_text = re.sub('<.*?>', '', conversation_text)

                two_grams = ngram_helper(conversation_text, 2)
                three_grams = ngram_helper(conversation_text, 3)
                four_grams = ngram_helper(conversation_text, 4)
                                
                word_list = nltk.word_tokenize(conversation_text)
                filtered_words = [word for word in word_list if word not in stopwords.words('english') and word.isalnum()]                
                lemmatized_words = [(lmtz.lemmatize(word)).encode('ascii', 'ignore') for word in filtered_words]

                lemmatized_words = lemmatized_words + two_grams
                lemmatized_words = lemmatized_words + three_grams
                lemmatized_words = lemmatized_words + four_grams
 
                category = calculate_probability(lemmatized_words, category_count_map, word_category_map, category_wordcount_map, vocabulary)
                print "writing " + str(conversation_id) + "," + category
                fileWriter.write(str(conversation_id) + "," + category + "\n")
                
            except Exception, e:
                print "Error in test_naive_bayes"
                #print e
                    
    fileWriter.close()
    
    
#Reads the output for the training data
def readOutput():
    #contains number of text in a particular category, <category, textcount>
    category_count_map = dict()
    #contains <conversation_id, category>
    conversationID_category_map = dict()
    
    with open('train_output.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            try:
                
                conversation_id = int(row[0])
                category = row[1]                
                if category in category_count_map:
                    category_count_map[category] = category_count_map[category] + 1
                else:
                    category_count_map[category] = 1                
                conversationID_category_map[conversation_id] = category
 

                
            except Exception,e:
                print e
                
    return category_count_map, conversationID_category_map        


def main():
    category_count_map, conversationID_category_map = readOutput()
    word_category_map, category_wordcount_map, vocabulary = readInput(category_count_map, conversationID_category_map)
    vocabularyCount = len(vocabulary)
    test_naive_bayes(category_count_map, word_category_map, category_wordcount_map, vocabularyCount)



    
if __name__ == '__main__':
    main()

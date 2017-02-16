'''
Created on Feb 7, 2017

@author: eanuama
'''

import csv
from collections import defaultdict


#This takes the output of 5 different runs of kNN and performs voting to predict the category. It's kind of ensemble.
def main():

#Takes the result of 5 different runs of kNN.
    src1 = "McGillResult\\src1\\result.csv"
    src2 = "McGillResult\\src2\\result.csv"
    src3 = "McGillResult\\src3\\result.csv"
    src4 = "McGillResult\\src4\\result.csv"
    src5 = "McGillResult\\src5\\result.csv"
    
    category_map =defaultdict()
    
    result1_map = dict()
    result2_map = dict()
    result4_map = dict()
    result5_map = dict()
    
    with open(src1, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            result1_map[row[0]] = row[1]
            category_map[row[1]] = 0
            


    with open(src2, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            result2_map[row[0]] = row[1]

            
    with open(src4, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            result4_map[row[0]] = row[1]
    


    with open(src5, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            result5_map[row[0]] = row[1]
    
    
            
       
    fileWriter = open("final_result.csv", "wb")
            
    with open(src3, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            category_map[result1_map[row[0]]] = category_map[result1_map[row[0]]] + 1  
            category_map[result2_map[row[0]]] = category_map[result2_map[row[0]]] + 1
            category_map[result4_map[row[0]]] = category_map[result4_map[row[0]]] + 1
            category_map[result5_map[row[0]]] = category_map[result5_map[row[0]]] + 1
            
            category_map[row[1]] = category_map[row[1]] + 1
                                                
            most_probable_category = sorted(category_map, key=category_map.get, reverse=True)
            fileWriter.write(row[0] + "," + most_probable_category[0] + "\n")
            print row[0] + "," + most_probable_category[0] + "\n"
            
            category_map_cleaner(category_map)

    fileWriter.close()
            

def category_map_cleaner(category_map):
    for key in category_map.keys():
        category_map[key] = 0
    return category_map

            
        
if __name__ == '__main__':
    main()
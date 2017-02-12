#!/usr/bin/env python

import csv
import numpy
import nltk
import re
import string

input_train = open("train_input.csv")
data_input = csv.reader(input_train, delimiter=',')
next(data_input)  # skip header row

output_train = open("train_output.csv")
data_output = csv.reader(output_train, delimiter=',')
next(data_output)  # skip header row

input_test = open("test_input.csv")
test_input = csv.reader(input_test, delimiter=',')
next(test_input)  # skip header row

output_test = open("test_output.csv", "w")
test_output = csv.writer(output_test)

topics = ['hockey', 'movies', 'nba', 'news', 'nfl', 'politics', 'soccer', 'worldnews']
tags = re.compile('<.*?>')
all_input, all_output, unique_words = [], [], []
class_freq = [0]*8

for r in data_input:
    id = r[0]
    parsed_text = re.sub(tags, '', r[1])
    p = parsed_text.translate(None, string.punctuation)
    words = nltk.word_tokenize(p)
    unique_words.extend(words)
    all_input.append(words)

for r in data_output:
    id2 = int(r[0])
    label = topics.index(r[1])
    class_freq[label] += 1
    all_output.append(label)
print class_freq
total = sum(class_freq)

unique_words = set(unique_words)
dicts, all_data = [], []
for i in range(8):
    dicts.append({w:0 for w in unique_words})

for pair in zip(all_input, all_output):
    d = dicts[pair[1]]  #get word frequency dictionary for current label
    for w in pair[0]:
        d[w] += 1
    #all_data.append(pair)
#for x in dicts:
#    print x


def compute_probs(words, i):
    p_x_given_y = 1.0
    d_i = dicts[i]
    num_i = class_freq[i]
    prob_i = float(num_i) / float(total)
    for w in words:
        if w not in unique_words:  # word doesn't exist in training set
            d_i[w] = 0
        p_x_given_y *= float(d_i[w] + 1) / float(num_i + 2)
        #print p_x_given_y, d_i[w]
    #print p_x_given_y, prob_i
    return p_x_given_y * prob_i

test_labels = []
j = 0
for r in test_input:
    parsed_text = re.sub(tags, '', r[1])
    p = parsed_text.translate(None, string.punctuation)
    words = nltk.word_tokenize(p)
    probs = []
    for i in range(8):
        p_y_given_x = compute_probs(words, i)  # assume label is i, compute P(x_i | y=i) * P(y=i)
        probs.append(p_y_given_x)
    #print probs
    label_index = probs.index(max(probs))
    #print label_index
    test_labels.append([j, topics[label_index]])
    j += 1

#print test_labels[:100]
test_output.writerows(test_labels)










import pandas as pd
import pickle
import numpy as np

df = pd.read_csv('./data/mag_paper_data/citation_graph.tsv', sep="\t")
id_match = pd.read_csv('./data/mag_paper_data/id_paperId.tsv', sep="\t")
train = pd.read_csv('./trainset.csv', dtype=str)
valid = pd.read_csv('./validset.csv', dtype=str)
test = pd.read_csv('./testset.csv', dtype=str)

dict1 = {}
wordset = train + valid + test

for i in wordset:
    for sentence in i['Abstract']:
        for word in sentence:
            if 	word not in dict1:
                dict1[word] = 1
            else:
                dict1[word] = dict1[word]+1
rare_word = []
for i in iter(dict1):
    if dict1[i] < 10:
        rare_word.append(i)
for i in rare_word:
    del dict1[i]
    
feature = np.zeros((len(wordset),len(dict1)))
for i in range(0,len(wordset)):
    paper = id_match[id_match['Id'] == dataset['Id'][i]]
    if len(paper) == 0:
        feature[i,0] = 0
        feature[i,1] = 0
        continue
    feature[i,0] = len(df[df['PaperId'] == paper.iloc[0]['PaperId']])
    feature[i,1] = len(df[df['CitedPaperId'] == paper.iloc[0]['PaperId']])

dict1_keys = dict1.keys()
for i,j in zip(wordset,range(0,len(wordset))):
    for sentence in i['Abstract']:
        for word in sentence:
            if word not in dict1:
                continue
            feature[j,list(dict1_keys).index(word)] = feature[j,list(dict1_keys).index(word)] + 1

with open('data/feature.pkl','wb') as f:
    pickle.dump(feature, f)
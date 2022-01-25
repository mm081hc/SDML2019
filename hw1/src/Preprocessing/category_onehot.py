import pandas as pd
import pickle
import numpy as np

dataset = pd.read_csv('./data/task2_trainset.csv', dtype=str)
dataset.drop('Title',axis=1,inplace=True)
dataset.drop('Abstract',axis=1,inplace=True)
dataset.drop('Created Date',axis=1, inplace=True)
dataset.drop('Authors',axis=1,inplace=True)
trainset, validset = train_test_split(dataset, test_size=0.1, random_state=42)
testset = pd.read_csv('./data/task2_public_testset.csv', dtype=str)
testset.drop('Title',axis=1,inplace=True)
testset.drop('Abstract',axis=1,inplace=True)
testset.drop('Created Date',axis=1, inplace=True)
testset.drop('Authors',axis=1,inplace=True)

cate_dict = {}

data = pd.concat([trainset,validset,testset],axis=0, ignore_index = True)
for i in data.iterrows():
    cate = i[1]['Categories'].split('/')
    for j in cate:
        cate_dict[j.split('.')[0]] = 1

cate_feature = np.zeros((len(data),len(cate_dict)))
for i,k in zip(data.iterrows(),range(len(data))):
    cate = i[1]['Categories'].split('/')
    for j in cate:
        cate_feature[k,list(cate_dict.keys()).index(j.split('.')[0])] = 1


with open('data/cate_feature.pkl','wb') as f:
    pickle.dump(cate_feature, f)
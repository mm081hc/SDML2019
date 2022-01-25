import numpy as np
import pandas as pd
import sys, os
from sklearn.model_selection import train_test_split
from bert_serving.client import BertClient
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import trange, tqdm

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)
from transformers.modeling_xlnet import *
from transformers.modeling_utils import SequenceSummary

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


NUM_CAT = 0

def get_cited_num():
    cite_graph = pd.read_csv('mag_paper_data/citation_graph.tsv', sep='\t').values
    pid2abstract = pd.read_csv('mag_paper_data/paper_title_abstract.tsv', sep='\t').values
    pid2abstract = dict(zip(pid2abstract[:, 0], pid2abstract[:, 2]))

    pid2citednum = dict()

    for cited_id in cite_graph[:, 1]:
        if(cited_id not in pid2abstract):
            continue
        if(cited_id not in pid2citednum):
            pid2citednum[cited_id] = 1
        else:
            pid2citednum[cited_id] += 1

    return pid2citednum

def get_cited_data(dataId, pid2citednum, save=False):
    cite_graph = pd.read_csv('mag_paper_data/citation_graph.tsv', sep='\t').values
    id2pid = pd.read_csv('mag_paper_data/id_paperId.tsv', sep='\t').values
    id2pid = dict(zip(id2pid[:, 0], id2pid[:, 1]))
    pid2abstract = pd.read_csv('mag_paper_data/paper_title_abstract.tsv', sep='\t').values
    pid2abstract = dict(zip(pid2abstract[:, 0], pid2abstract[:, 2]))

    pid2citenum = dict()
    data_mask = []
    data_abstract = []

    for i, did in enumerate(dataId):
        if(did not in id2pid or id2pid[did] not in pid2abstract):
            continue
        cite_papers = cite_graph[cite_graph[:, 0]==id2pid[did]][:, 1]
        min_cite_num = float("inf")
        min_cite_pid = None
        for pid in cite_papers:
            if(pid not in pid2citednum):
                continue
            cite_num = pid2citednum[pid]

            if(cite_num < min_cite_num):
                min_cite_num = cite_num
                min_cite_pid = pid

        if(min_cite_pid == None):
            continue

        data_abstract.append(pid2abstract[min_cite_pid])
        data_mask.append(i)

    print('Total num = ', len(data_mask))

    if(save):
        bc = BertClient(check_length=False)
        print('Start encoding...')
        cite_vec = bc.encode(data_abstract)
    else:
        cite_vec = []

    return cite_vec, data_mask



def load_data(save=False, xlnet=False):
    dataset = pd.read_csv('data/task2_trainset.csv', dtype=str).values
    testset = pd.read_csv('data/task2_public_testset.csv', dtype=str).values


    trainset, validset = train_test_split(dataset, test_size=0.1, random_state=42)

    categories = dataset[:, 4]
    category_list = []

    for cat in categories:
        cat = cat.split('/')
        category_list += cat

    category_list = np.unique(category_list)

    # NUM_CAT += len(category_list)

    cat2id = dict(zip(category_list, range(len(category_list))))

    if(xlnet):
        sequence_summary = SequenceSummary(XLNetConfig)
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetModel.from_pretrained('xlnet-base-cased')
        train_vec, valid_vec = [], []
        for s in trainset[:, 2]:
            input_ids = tokenizer.encode(s, add_special_tokens=True, max_length=512)
            input_ids = [0]*(300-len(input_ids))+input_ids
            train_vec.append(input_ids)
        out = model(torch.LongTensor(train_vec))[0]
        train_vec = sequence_summary(out).tolist()
            
        for s in validset[:, 2]:
            input_ids = tokenizer.encode(s, add_special_tokens=True, max_length=512)
            input_ids = [0]*(300-len(input_ids))+input_ids
            valid_vec.append(input_ids)
        out = model(torch.LongTensor(valid_vec))[0]
        valid_vec = sequence_summary(out).tolist()

        np.save('data/train_vec_xlnet.npy', np.array(train_vec))
        np.save('data/valid_vec_xlnet.npy', np.array(valid_vec))

        test_vec = []
    else:
        if(save):
            bc = BertClient(check_length=False)

            print('Start encoding training vector.')
            train_vec = bc.encode(trainset[:, 2].tolist())
            print('Start encoding validation vector.')
            valid_vec = bc.encode(validset[:, 2].tolist())

            print('Successfully encode vectors.')
            np.save('data/train_vec_sci.npy', np.array(train_vec))
            np.save('data/valid_vec_sci.npy', np.array(valid_vec))
            test_vec = np.load('data/test_vec_sci.npy')


        else:
            train_vec = np.load('data/train_vec_tuned.npy')
            valid_vec = np.load('data/valid_vec_tuned.npy')
            test_vec = np.load('data/test_vec.npy')



    # pid2citednum = get_cited_num()
    # train_cite_vec, train_mask = get_cited_data(trainset[:, 0], pid2citednum, save)
    # valid_cite_vec, valid_mask = get_cited_data(validset[:, 0], pid2citednum, save)

    # if(save):
    #     np.save('data/train_cite_vec.npy', train_cite_vec)
    #     np.save('data/valid_cite_vec.npy', valid_cite_vec)
    # else:
    #     train_cite_vec = np.load('data/train_cite_vec.npy')
    #     valid_cite_vec = np.load('data/valid_cite_vec.npy')

    # trainset, validset = trainset[train_mask], validset[valid_mask]
    # train_vec, valid_vec = train_vec[train_mask], valid_vec[valid_mask]

    # print(len(trainset), len(train_vec))

    train_cite_vec, valid_cite_vec = np.zeros(len(trainset)), np.zeros(len(validset))


    label2id = {"THEORETICAL":0, "ENGINEERING":1, "EMPIRICAL":2, "OTHERS":3}
    train_label, valid_label = [], []
    train_cat, valid_cat, test_cat = [], [], []

    the_vec = np.zeros(len(train_vec[0]))
    eng_vec = np.zeros(len(train_vec[0]))
    emp_vec = np.zeros(len(train_vec[0]))

    for i, (cat, label) in enumerate(trainset[:, [4, -1]]):
        label_vec = [0]*4
        cat_vec = [0]*len(category_list)
        for l in label.split(' '):
            label_vec[label2id[l]] = 1
            if(l == 'THEORETICAL'):
                the_vec += train_vec[i]
            elif(l == 'ENGINEERING'):
                eng_vec += train_vec[i]
            elif(l == 'EMPIRICAL'):
                emp_vec += train_vec[i]

        train_label.append(label_vec)

        for c in cat.split('/'):
            cat_vec[cat2id[c]] = 1
        train_cat.append(cat_vec)


    for cat, label in validset[:, [4, -1]]:
        label_vec = [0]*4
        cat_vec = [0]*len(category_list)
        for l in label.split(' '):
            label_vec[label2id[l]] = 1
        valid_label.append(label_vec)

        for c in cat.split('/'):
            if(c not in cat2id):
                continue
            cat_vec[cat2id[c]] = 1
        valid_cat.append(cat_vec)


    for cat in testset[:, 4]:
        cat_vec = [0]*len(category_list)

        for c in cat.split('/'):
            if(c not in cat2id):
                continue
            cat_vec[cat2id[c]] = 1
        test_cat.append(cat_vec)





    ave_vec = np.concatenate([[the_vec/len(train_vec)], [eng_vec/len(train_vec)], [emp_vec/len(train_vec)]])

    sim_train = np.dot(train_vec, ave_vec.T)
    sim_valid = np.dot(valid_vec, ave_vec.T)

    # train_cat = np.concatenate([train_cat, train_cite_vec], axis=1)
    # valid_cat = np.concatenate([valid_cat, valid_cite_vec], axis=1)

    train_abstract_len = np.array([len(m.split('/')) for m in trainset[:, 3]])
    valid_abstract_len = np.array([len(m.split('/')) for m in validset[:, 3]])
    abstract_mean = np.mean(train_abstract_len)
    abstract_std = np.std(train_abstract_len)

    train_abstract_len = (train_abstract_len - abstract_mean) / (abstract_std+1e-20)
    valid_abstract_len = (valid_abstract_len - abstract_mean) / (abstract_std+1e-20)

    print(np.array([train_abstract_len]))

    # train_cat = np.concatenate([train_cat, train_abstract_len.reshape(-1, 1)], axis=1)
    # valid_cat = np.concatenate([valid_cat, valid_abstract_len.reshape(-1, 1)], axis=1)


    
    return torch.FloatTensor(train_vec), torch.FloatTensor(valid_vec), torch.FloatTensor(test_vec), torch.FloatTensor(train_label), torch.FloatTensor(valid_label), torch.FloatTensor(train_cat), torch.FloatTensor(valid_cat), torch.FloatTensor(test_cat)

def load_test_data(save=False):
    bc = BertClient(check_length=False)
    dataset = pd.read_csv('data/task2_public_testset.csv', dtype=str).values
    print('Start encoding test vector.')
    test_vec = bc.encode(dataset[:, 2].tolist(), )
    print('Finished.')
    test_vec = np.array(test_vec)
    np.save('data/test_vec.npy', np.array(test_vec))

class F1():
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.name = 'F1'

    def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0

    def update(self, predicts, groundTruth):
        # for i, predict in enumerate(predicts):
            # if(torch.max(predict[:3]).data.item() > 0.6):
                # predicts[i][3] = 0
            # if(torch.max(predict[:3]).data.item() < self.threshold):
                # predicts[i][3] = 1
        #     # print(torch.max(predict).data.item(), torchmax(predict).data.item() <= 0.5)
            # if(torch.max(predict).data.item() <= self.threshold):
                # predicts[i][torch.argmax(predict)] = 1

        predicts = predicts > self.threshold
        self.n_precision += torch.sum(predicts).data.item()
        self.n_recall += torch.sum(groundTruth).data.item()
        self.n_corrects += torch.sum(groundTruth.type(torch.uint8) * predicts).data.item()

    def get_score(self):
        recall = self.n_corrects / self.n_recall
        precision = self.n_corrects / (self.n_precision + 1e-20)
        return 2 * (recall * precision) / (recall + precision + 1e-20)

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)

class AbstractDataset(Dataset):
    def __init__(self, vec, cat, label):
        self.vec = vec
        self.label = label
        self.cat = cat
        # self.cite_vec = cite_vec
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        return self.vec[index], self.cat[index], self.label[index]

class TestDataset(Dataset):
    def __init__(self, vec, cat):
        self.vec = vec
        self.cat = cat
    def __len__(self):
        return len(self.cat)
    def __getitem__(self, index):
        return self.vec[index], self.cat[index]


class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        # self.l1 = nn.Linear(1165, 4)
        self.l1 = nn.Linear(909, 4)
        # self.l1 = nn.Linear(768, 4)
        self.dropout = nn.Dropout(p=0.1)
        # self.l1 = nn.Linear(768, 4)

    def forward(self, x, x_cat):
        # print(x.size(), x_cat.size())
        # x = torch.cat([x, x_cat], dim=1)
        # print(x.size())
        # x_cat = self.l2(x_cat)
        # x = self.l3(x)
        x = torch.cat([x, x_cat], dim=1)
        # x = self.dropout(x)
        x = self.l1(x)
        x = torch.sigmoid(x)
        return x

model = simpleNet()
model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=2e-4)
criteria = torch.nn.BCELoss()


def run_epoch(dataset, training):
    if training:
        description = 'Train'
        shuffle = True
    else:
        description = 'Valid'
        shuffle = False

    dataloader = DataLoader(dataset=dataset,
                            batch_size=32,
                            shuffle=shuffle)

    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
    
    loss = 0

    if(1):
        f1_score = F1()
        for i, (x, x_cat, y) in trange:
            x = x.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)
            logits = model(x, x_cat)
            l_loss = criteria(logits, y)
            if(training):
                opt.zero_grad()
                l_loss.backward()
                opt.step()

            loss += l_loss.item()
            f1_score.update(logits.cpu(), y.cpu())
        return loss / len(dataloader), f1_score.get_score()
    # else:
    #     f1_max = 0
    #     for t0 in np.arange(0.3, 0.6, 0.02):
    #         for t1 in np.arange(0.3, 0.6, 0.02):
    #             for t2 in np.arange(0.3, 0.6, 0.02):
    #                 for t3 in np.arange(0.3, 0.6, 0.02):
    #                     f1_score = F1(threshold=torch.Tensor([t0, t1, t2, t3]))
    #                     for i, (x, x_cat, y) in trange:
    #                         x = x.to(device)
    #                         x_cat = x_cat.to(device)
    #                         y = y.to(device)
    #                         logits = model(x, x_cat)
    #                         l_loss = criteria(logits, y)
    #                         if(training):
    #                             opt.zero_grad()
    #                             l_loss.backward()
    #                             opt.step()

    #                         loss += l_loss.item()
    #                         f1_score.update(logits.cpu(), y.cpu())
    #                         if(f1_score.get_score() > f1_max):
    #                             f1_max = f1_score.get_score()

    #     return loss / len(dataloader), f1_max

def save():
    if not os.path.exists('model_service'):
        os.makedirs('model_service')
    torch.save(model.state_dict(), 'model_service/model.pkl')


def SubmitGenerator(prediction, sampleFile, public=True, filename='prediction.csv'):
    """
    Args:
        prediction (numpy array)
        sampleFile (str)
        public (boolean)
        filename (str)
    """
    sample = pd.read_csv(sampleFile)
    submit = {}
    submit['order_id'] = list(sample.order_id.values)
    redundant = len(sample) - prediction.shape[0]
    if public:
        submit['THEORETICAL'] = list(prediction[:,0]) + [0]*redundant
        submit['ENGINEERING'] = list(prediction[:,1]) + [0]*redundant
        submit['EMPIRICAL'] = list(prediction[:,2]) + [0]*redundant
        submit['OTHERS'] = list(prediction[:,3]) + [0]*redundant
    else:
        submit['THEORETICAL'] = [0]*redundant + list(prediction[:,0])
        submit['ENGINEERING'] = [0]*redundant + list(prediction[:,1])
        submit['EMPIRICAL'] = [0]*redundant + list(prediction[:,2])
        submit['OTHERS'] = [0]*redundant + list(prediction[:,3])
    df = pd.DataFrame.from_dict(submit) 
    df.to_csv(filename,index=False)


def main():
    train_vec, valid_vec, test_vec, train_label, valid_label, train_cat, valid_cat, test_cat = load_data(save=True)    

    trainData = AbstractDataset(train_vec, train_cat, train_label)
    validData = AbstractDataset(valid_vec, valid_cat, valid_label)
    testData = TestDataset(test_vec, test_cat)

    print('Start training...')

    patience = 0
    min_loss = float("inf")
    min_loss_f1 = 0

    max_epoch = 200
    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        loss, f1 = run_epoch(trainData, True)
        print('Training loss, f1 = ', loss, f1)
        loss, f1 = run_epoch(validData, False)
        if(loss < min_loss):
            min_loss = loss
            patience = 0
            min_loss_f1 = f1
            save()
        else:
            patience += 1
        print('Validation loss, f1 = ', loss, f1)
        if(patience > 4):
            break

    print('f1 = ', min_loss_f1)

    # load_test_data()

    model.load_state_dict(torch.load('model_service/model.pkl'))
    model.train(False)
    dataloader = DataLoader(dataset=testData,
                                batch_size=128,
                                shuffle=False,)
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    prediction = []
    for i, (x, x_cat) in trange:
        o_labels = model(x.to(device), x_cat.to(device))
        # o_labels = o_labels>0.4
        prediction.append(o_labels.to('cpu'))

    prediction = torch.cat(prediction).detach().numpy().astype(float)

    SubmitGenerator(prediction, 
                'data/task2_sample_submission.csv',
                True, 
                'submission_bert.csv')

if __name__ == '__main__':
    main()










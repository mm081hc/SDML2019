import torch
import numpy as np
import scipy.sparse as sp
import pickle
import pandas as pd


def load_data():
    
    with open('../../feature_word.pkl','rb') as f:
        idx_features = pickle.load(f)
    with open('../../labels.pkl','rb') as f:
        labels = pickle.load(f)
    with open('../../adj.pkl','rb') as f:
        edges_unordered = pickle.load(f)
    with open('../../dataset.pkl','rb') as f:
        dataset = pickle.load(f)
   
    labels = np.array(labels)
    features = sp.csr_matrix(idx_features, dtype=np.float32)
    # build graph
    idx = np.array(dataset['Id'], dtype='a32')        
    idx_map = {j: i for i, j in enumerate(idx)}

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype='a32').reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.int32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(6300)
    idx_val = range(6301, 7000)
    idx_test = range(7001, 27000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.FloatTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = torch.exp(output)>0.33
    preds = preds.detach().numpy().astype(int)
    labels = labels.detach().numpy().astype(int)
    TP = 0
    FP = 0
    FN = 0
    for i in range(4):
        for j in range(len(labels)):
            if labels[j][i] == preds[j][i] and labels[j][i] == 1:
                TP = TP + 1
            if labels[j][i] != preds[j][i] and labels[j][i] == 1:
                FP = FP + 1
            if labels[j][i] != preds[j][i] and labels[j][i] == 0:
                FN = FN + 1
    prec = TP/(TP+FP+1e-20)
    rec =  TP/(TP+FN+1e-20)
    f1 = 2*prec*rec/(prec+rec+1e-20)
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

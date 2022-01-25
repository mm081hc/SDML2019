import pandas as pd
import pickle
import numpy as np

df = pd.read_csv('./data/mag_paper_data/citation_graph.tsv', sep="\t")
id_match = pd.read_csv('./data/mag_paper_data/id_paperId.tsv', sep="\t")

adj = pd.DataFrame(columns=['paper', 'cited'])

a = df[df['PaperId'].isin(id_match['PaperId'])]
b = a[a['CitedPaperId'].isin(id_match['PaperId'])]
adj = np.empty((len(b),2),dtype='a32')
for i,j in zip(b.iterrows(),range(0,len(b))):
    adj[j,0] = str(id_match[id_match['PaperId'] == i[1]['PaperId']].iloc[0]['Id'])
    adj[j,1] = str(id_match[id_match['PaperId'] == i[1]['CitedPaperId']].iloc[0]['Id'])

with open('data/adj.pkl','wb') as f:
    pickle.dump(adj, f)
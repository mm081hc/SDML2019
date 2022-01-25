import pandas as pd
import numpy as np
import sys

def ensemble_prob(outfile="ans_ensemble_prob.csv", inputfile1="xlnet_prob.csv", inputfile2="bert_prob.csv", inputfile3="ans_xlnet04_r.csv"):
	with open(inputfile1, 'r') as f:
		df1 = pd.read_csv(inputfile1)
	with open(inputfile2, 'r') as f:
		df2 = pd.read_csv(inputfile2)

	c1 = np.where((df1['THEORETICAL'].values + df2['THEORETICAL'].values)/2 > 0.4, 1, 0)
	c2 = np.where((df1['ENGINEERING'].values + df2['ENGINEERING'].values)/2 > 0.4, 1, 0)
	c3 = np.where((df1['EMPIRICAL'].values + df2['EMPIRICAL'].values)/2 > 0.4, 1, 0)

	with open(outfile, 'w') as f:
		f.write('order_id,THEORETICAL,ENGINEERING,EMPIRICAL,OTHERS\n')
		for i in range(40000):
			pred = [c1[i], c2[i], c3[i], 0]
			if(np.sum(pred) == 0 and i < 20000):
				pred = [c1[i], c2[i], c3[i], 1]

			f.write('T' + '%05d'%(i+1) + ',' + ','.join(map(str,pred)) + '\n')

if(len(sys.argv) > 1):
	ensemble_prob(sys.argv[1])
else:
	ensemble_prob()
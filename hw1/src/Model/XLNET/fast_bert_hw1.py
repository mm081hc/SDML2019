import numpy as np
import pandas as pd

label2id = {"THEORETICAL":0, "ENGINEERING":1, "EMPIRICAL":2, "OTHERS":3}

def create_file(path):
    outlist = []
    f = open(path.replace('.csv', '_fast_bert.csv'), 'w')
    outlist.append(['id','text','THEORETICAL','ENGINEERING','EMPIRICAL','OTHERS'])
    f.write('id,text,THEORETICAL,ENGINEERING,EMPIRICAL,OTHERS\n')
    data = pd.read_csv(path, encoding='utf8', dtype=str).values
    for i, d in enumerate(data):
        label = d[-1].split(' ')
        d[1] = d[1].replace(',', '.')
        onehot_label = ['0']*len(label2id)
        for l in label:
            onehot_label[label2id[l]] = '1'

        outlist.append(list(d[:-1])+onehot_label)
        f.write(','.join(d[:-1])+','+','.join(onehot_label)+'\n')

    # print(outlist)

    df = pd.DataFrame(outlist)
    df.to_csv(index=False)
    np.savetxt(path.replace('.csv', '_fast_bert.csv'), outlist, delimiter=",", fmt='%s')

def train():
    from fast_bert.data_cls import BertDataBunch
    from fast_bert.learner_cls import BertLearner
    from fast_bert.metrics import accuracy, F1
    import logging
    import torch
    databunch = BertDataBunch('data/', 'data/',
                          tokenizer='xlnet-large-uncased',
                          train_file='trainset_fast_bert.csv',
                          val_file='validset_fast_bert.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col=['THEORETICAL','ENGINEERING','EMPIRICAL','OTHERS'],
                          batch_size_per_gpu=6,
                          max_seq_length=100,
                          multi_gpu=True,
                          multi_label=True,
                          model_type='xlnet')

    logger = logging.getLogger()
    device_cuda = torch.device("cuda:0")
    metrics = [{'name': 'f1', 'function': F1}]

    learner = BertLearner.from_pretrained_model(
                            databunch,
                            pretrained_path='xlnet-large-uncased',
                            metrics=metrics,
                            device=device_cuda,
                            logger=logger,
                            output_dir='out_fast_bert',
                            finetuned_wgts_path=None,
                            warmup_steps=500,
                            multi_gpu=True,
                            is_fp16=False,
                            multi_label=True,
                            logging_steps=197)

    learner.fit(epochs=8,
            lr=6e-5,
            validate=True,  # Evaluate the model after each epoch
            schedule_type="warmup_cosine",
            optimizer_type="lamb")


    learner.save_model()
def test(model_path='out_fast_bert/model_out_xlnet70', testfile='data/testset.csv', outfile='xlnet_prob.csv'):
    from fast_bert.prediction import BertClassificationPredictor
    import pandas as pd
    predictor = BertClassificationPredictor(
        model_path=model_path,
	label_path='data/',
	multi_label=True,
	model_type='xlnet',
	do_lower_case=False)
    df = pd.read_csv(testfile)

    print(df)
    texts = df['Abstract'].values
    multi_pred = predictor.predict_batch(list(texts))
    with open(outfile, 'w') as f:
        f.write('order_id,THEORETICAL,ENGINEERING,EMPIRICAL,OTHERS\n')
        for i in range(1, 40001):
            if(i <= 20000):
                #pred = predictor.predict(texts[i-1])
                pred = multi_pred[i-1]
                print(pred)
                pred_w = [0, 0, 0, 0]
                pred_p = [0, 0, 0, 0]
                if(i % 1000 == 0):
                    print(i)
                for j in range(4):

                    pred_p[label2id[pred[j][0]]] = pred[j][1] #probability
                    if(pred[j][1] >= 0.4):
                        if(j == 0 and pred[j][0] == "OTHERS"):
                            pred_w[label2id[pred[j][0]]] = 1
                            break
                        elif(pred[j][0] == "OTHERS"):
                            continue
                        else:
                            pred_w[label2id[pred[j][0]]] = 1
                           
                    elif(pred[0][1] < 0.4):
                        pred_w[label2id[pred[j][0]]] = 1
                        break    
                f.write('T' + '%05d' % i + ',' + ','.join(map(str,pred_p)) + '\n')
            else:
                f.write("T%s,0,0,0,0\n" % i)

if __name__ == '__main__':
    

    train()
    # test()















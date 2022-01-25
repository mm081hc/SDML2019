from torch import nn
from model.Encoder import VanillaEncoder
from model.Decoder import VanillaDecoder
from model.Seq2Seq import Seq2Seq
from dataset.DataHelper import DataTransformer
from train import Trainer
from config import config


def main():
    data_transformer = DataTransformer(config.dataset_path, use_cuda=config.use_cuda)
    # share embedding
    embedding = nn.Embedding(data_transformer.vocab_size, config.encoder_embedding_size)

    vanilla_encoder = VanillaEncoder(vocab_size=data_transformer.vocab_size,
                                     embedding_size=config.encoder_embedding_size,
                                     output_size=config.encoder_output_size,
                                     embedding=embedding)

    vanilla_decoder = VanillaDecoder(hidden_size=config.decoder_hidden_size,
                                     output_size=data_transformer.vocab_size,
                                     max_length=data_transformer.max_length,
                                     teacher_forcing_ratio=config.teacher_forcing_ratio,
                                     sos_id=data_transformer.SOS_ID,
                                     use_cuda=config.use_cuda,
                                     embedding=embedding)
                                     
    if config.use_cuda:
        vanilla_encoder = vanilla_encoder.cuda()
        vanilla_decoder = vanilla_decoder.cuda()

    seq2seq = Seq2Seq(encoder=vanilla_encoder,
                      decoder=vanilla_decoder)

    trainer = Trainer(seq2seq, data_transformer, config.learning_rate, config.use_cuda,checkpoint_name="task2-2_2gru_share_embedding_2epoch.pt")
    trainer.load_model()

    with open('./dataset/hw2.1-1_testing_data.txt', 'r') as testset:
        results = []
        seq_list = []
        i = 0
        for seq in testset:
            seq = seq.strip('\n')
            seq_list.append(seq)
        for j in range(0, 500):
            #print(seq_list[j*10:(j*10+10)])
            results += trainer.evaluate(seq_list[(j*100):(j*100+100)])
            print(j, results[j*100])
            #print(results)
        
        #for seq in testset:
            #seq = seq.strip('\n')
            #results.append(trainer.evaluate(seq)[0])
            #if(i > 10):
            #    break
            #if(i % 1000 == 0):
            #    print(results[i])
            #i += 1

    with open('./dataset/task2-1_using2-2.txt', 'w') as f:
        for s in results:
            f.write(s + '\n')

if __name__ == "__main__":
    main()

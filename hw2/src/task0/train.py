import torch
import random

from model.Encoder import VanillaEncoder
from model.Decoder import VanillaDecoder
from model.Seq2Seq import Seq2Seq
from dataset.DataHelper import DataTransformer
from config import config


class Trainer(object):

    def __init__(self, model, data_transformer, learning_rate, use_cuda, device,
                 checkpoint_name=config.checkpoint_name,
                 teacher_forcing_ratio=config.teacher_forcing_ratio):

        self.model = model

        # record some information about dataset
        self.data_transformer = data_transformer
        self.vocab_size = self.data_transformer.vocab_size
        self.PAD_ID = self.data_transformer.PAD_ID
        self.use_cuda = use_cuda
        self.device = device

        # optimizer setting
        self.learning_rate = learning_rate
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.NLLLoss(ignore_index=self.PAD_ID, size_average=True)
        
        #vaild data
        self.valid_data = [i.strip("\n") for i in open("dataset/200_vocab_dataset.txt", "r")][19000:]
        self.checkpoint_name = checkpoint_name

    def train(self, num_epochs, batch_size, pretrained=False):

        if pretrained:
            self.load_model()

        step = 0

        for epoch in range(0, num_epochs):
            mini_batches = self.data_transformer.mini_batches(batch_size=batch_size)
            for input_batch, target_batch in mini_batches:
                self.optimizer.zero_grad()
                decoder_outputs, decoder_hidden = self.model(input_batch, target_batch)

                # calculate the loss and back prop.
                cur_loss = self.get_loss(decoder_outputs, target_batch[0])

                # logging
                step += 1
                if step % 500 == 0:
                    self.save_model()
                    print("Step:", step, "loss of char: ", cur_loss.data, "valid_acc: ", self.valid_acc())
                    
                cur_loss.backward()

                # optimize
                self.optimizer.step()

        self.save_model()

    def masked_nllloss(self):
        # Deprecated in PyTorch 2.0, can be replaced by ignore_index
        # define the masked NLLoss
        weight = torch.ones(self.vocab_size)
        weight[self.PAD_ID] = 0
        if self.use_cuda:
            weight = weight.to(self.deivce)
        return torch.nn.NLLLoss(weight=weight).to(self.device)

    def get_loss(self, decoder_outputs, targets):
        b = decoder_outputs.size(1)
        t = decoder_outputs.size(0)
        targets = targets.contiguous().view(-1)  # S = (B*T)
        decoder_outputs = decoder_outputs.view(b * t, -1)  # S = (B*T) x V
        return self.criterion(decoder_outputs, targets)
    
    def valid_acc(self):
        acc = 0
    
        for i in self.valid_data:
            predict = self.evaluate(i)
            if(i == predict[0]):
                acc += 1
        acc = acc / len(self.valid_data)

        return acc

    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint_name)
        print("Model has been saved as %s.\n" % self.checkpoint_name)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name))
        print("Pretrained model has been loaded.\n")

    def tensorboard_log(self):
        pass

    def evaluate(self, words):
        # make sure that words is list
        if type(words) is not list:
            words = [words]

        # transform word to index-sequence
        eval_var = self.data_transformer.evaluation_batch(words=words)
        decoded_indices = self.model.evaluation(eval_var)
        results = []
        for indices in decoded_indices:
            indices = indices.tolist()
            results.append(self.data_transformer.vocab.indices_to_sequence(indices))
        return results


def main():
    data_transformer = DataTransformer(config.dataset_path, use_cuda=config.use_cuda)

    # define our models
    embedding = torch.nn.Embedding(data_transformer.vocab_size, config.encoder_embedding_size) # share embedding layer
    

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
                                     device=config.device,
                                     embedding=embedding
                                     )
    if config.use_cuda:
        vanilla_encoder = vanilla_encoder.cuda()
        vanilla_decoder = vanilla_decoder.cuda()


    seq2seq = Seq2Seq(encoder=vanilla_encoder,
                      decoder=vanilla_decoder)

    trainer = Trainer(seq2seq, data_transformer, config.learning_rate, config.use_cuda, config.device)
    trainer.train(num_epochs=config.num_epochs, batch_size=config.batch_size, pretrained=True)

if __name__ == "__main__":
    main()

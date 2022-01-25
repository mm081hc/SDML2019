import torch
# Define hyper parameter
use_cuda = True if torch.cuda.is_available() else False
device = 0
task = 2

# for dataset
dataset_path = 'dataset/hw2.1-2_train.tsv'

# for training
num_epochs = 1 # 12
batch_size = 128
learning_rate = 1e-3

# for model
encoder_embedding_size = 256
#encoder_output_size = 128
encoder_output_size = 256
decoder_hidden_size = encoder_output_size
teacher_forcing_ratio = .5
# max_length = 20

# for logging
checkpoint_name = 'task2-2_2gru_share_embedding_2epoch.pt'

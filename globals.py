# dimension of K(=Q), V
global d_k
global d_v

# number of heads in Multi-Head Attention
global n_heads

# FeedForward dimension
global d_ff

# d_time + feature_num
global d_model

# time2vec hidden size
global hidden_dim
hidden_dim = 10

# feature map embedding
global input_emb
input_emb = 6

global dropout
dropout = 0.4

# model criterion & optimizer
global criterion
global optimizer

# base hyper parameters
global epochs
global lr
global weight_decay
epochs = 50
lr = 1e-3
weight_decay = 0.

# data process parameters
global window_size
global predict_size
global stride
window_size = 7
predict_size = 2
stride = 2

import math
import torch
import numpy as np
import torch.nn as nn
import globals
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FEATURE_NUM = 6

def get_attn_subsequence_mask(seq):
    '''
    Generate mask for Decoder masked multi-head attention.

    seq: [batch_size, tgt_len, feature_num]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

class Time2Vec(nn.Module):
    def __init__(self, activation, hidden_dim, d_time):
        '''
        Embed time series data to keep time info.
        Reference: https://arxiv.org/pdf/1907.05321.pdf

        :param activation: sin/cos
        :param hidden_dim: custom
        '''
        super(Time2Vec, self).__init__()
        if activation == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos
        self.out_features = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, d_time, bias=False)
    def forward(self, x):
        x = x.float().to(device)
        # get all size
        batch_size = x.shape[0]
        sentence_len = x.shape[1]
        in_features = x.shape[2]
        self.w0 = nn.parameter.Parameter(torch.randn(batch_size, in_features, 1)).to(device)
        self.b0 = nn.parameter.Parameter(torch.randn(batch_size,sentence_len, 1)).to(device)
        self.w = nn.parameter.Parameter(torch.randn(batch_size, in_features, self.out_features - 1)).to(device)
        self.b = nn.parameter.Parameter(torch.randn(batch_size,sentence_len, self.out_features - 1)).to(device)
        v1 = self.activation(torch.matmul(x, self.w) + self.b)
        v2 = torch.matmul(x, self.w0) + self.b0
        v3 = torch.cat([v1, v2], -1)
        x = self.fc1(v3)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Calculate self-attention.

        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(globals.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, ):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(globals.d_model, globals.d_k * globals.n_heads, bias=False)
        self.W_K = nn.Linear(globals.d_model, globals.d_k * globals.n_heads, bias=False)
        self.W_V = nn.Linear(globals.d_model, globals.d_v * globals.n_heads, bias=False)
        self.fc = nn.Linear(globals.n_heads * globals.d_v, globals.d_model, bias=False)
        self.dropout = nn.Dropout(globals.dropout)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        Calculate multi-head attention.

        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        input_Q = input_Q.float()
        input_K = input_K.float()
        input_V = input_V.float()
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, globals.n_heads, globals.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, globals.n_heads, globals.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, globals.n_heads, globals.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, globals.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, globals.n_heads * globals.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        output = self.dropout(output + residual)
        return nn.LayerNorm(globals.d_model).to(device)(output), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(globals.d_model, globals.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(globals.d_ff, globals.d_model, bias=False)
        )
        self.dropout = nn.Dropout(globals.dropout)
    def forward(self, inputs):
        '''
        Calculate Add&Norm layer.

        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        output = self.dropout(output + residual)
        return nn.LayerNorm(globals.d_model).to(device)(output)
    
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        Encoder single unit.

        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: None (src_len always the same)
        '''
        # enc_outputs: [batch_size, src_len, d_time+feature_num], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, enc_layers, d_time, feature_num):
        super(Encoder, self).__init__()
        self.time_emb = Time2Vec('sin', globals.hidden_dim, d_time)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(enc_layers)])
        self.input_fc = nn.Linear(
            in_features=feature_num,
            out_features=globals.input_emb,
            bias=False
        )
    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, feature_num]
        '''
        enc_inputs = self.input_fc(enc_inputs.float()) # [batch_size, src_len, input_emb]
        time_emb_outputs = self.time_emb(enc_inputs) # [batch_size, src_len, d_time]
        enc_emb_outputs = torch.cat([enc_inputs, time_emb_outputs], dim=-1) # [batch_size, src_len, d_model]
        # enc_emb_outputs = time_emb_outputs 
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_emb_outputs, None)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, dec_layers, d_time, feature_num):
        super(Decoder, self).__init__()
        self.time_emb = Time2Vec('sin', globals.hidden_dim, d_time)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(dec_layers)])
        self.input_fc = nn.Linear(
            in_features=feature_num,
            out_features=globals.input_emb
        )
    def forward(self, dec_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len, 1]
        enc_intpus: [batch_size, src_len, feature_num]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_inputs = self.input_fc(dec_inputs.float()) # [batch_size, src_len, input_emb]
        time_emb_outputs = self.time_emb(dec_inputs) # [batch_size, src_len, d_time]
        dec_emb_outputs = torch.cat([dec_inputs, time_emb_outputs], dim=-1) # [batch_size, src_len, new_d_model(d_time+feature_num)]
        # dec_emb_outputs = time_emb_outputs
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt(dec_self_attn_subsequence_mask, 0).to(device) # [batch_size, tgt_len, tgt_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_emb_outputs, enc_outputs, dec_self_attn_mask, None)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self, d_time, enc_layers, dec_layers, d_k, d_v, d_ff, n_heads):
        super(Transformer, self).__init__()
        globals.d_k = d_k
        globals.d_v = d_v
        globals.d_ff = d_ff
        globals.n_heads = n_heads
        globals.d_model = d_time + FEATURE_NUM
        # globals.d_model = d_time
        assert d_k * n_heads == globals.d_model
        self.enc_feature = FEATURE_NUM
        self.dec_feature = 1
        self.encoder = Encoder(enc_layers, d_time, self.enc_feature).to(device)
        self.decoder = Decoder(dec_layers, d_time, self.dec_feature).to(device)
        self.projection = nn.Linear(globals.d_model, 1, bias=False).to(device)
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len, feature_num]
        dec_inputs: [batch_size, tgt_len, 1]
        '''
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, 1]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


import torch
import torch.nn as nn
import torch.nn.functional as F

import math




'''
Single attention head of multi-head attention.
'''
class Attention(nn.Module):

    def __init__(self, model_dim, Q_dim, K_dim, V_dim, device):
        super().__init__()

        self.query_matrix = nn.Linear(model_dim, Q_dim)
        self.key_matrix = nn.Linear(model_dim, K_dim)
        self.value_matrix = nn.Linear(model_dim, V_dim)
        self.device = device
    

    ## Q1(3 points): implement the scaled dot-product attention using softmax(QK^T / sqrt(K_dim))V.
    ## when calculating QK^T, masking the upper-triangle portion of QK^T with -inf when mask is set to True.
    def scaled_dot_product_attention(self, Q, K, V, mask):
        '''
        Q, K, V: linearly projected query, key and value. 
        Q.shape: (batch_size, seq_len, Q_dim); K.shape: (batch_size, seq_len, K_dim); V.shape: (batch_size, seq_len, V_dim);
        mask: a Boolean variable. mask is set to be True when calculating the masked multi-head attention in the decoder. 
        '''
        ## TO DO
        # att = torch.zeros(Q.shape[0], Q.shape[1], V.shape[-1]).to(self.device)

        QK = torch.bmm(Q, K.transpose(1, 2))  # shape: (batch_size, seq_len, seq_len)
        scaling_factor = torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32, device=self.device))
        QK_scaled = QK / scaling_factor
        if mask:
            mask = torch.tril(torch.ones(QK_scaled.size(-2), QK_scaled.size(-1), device=self.device)).bool()
            QK_scaled.masked_fill_(mask, -float('inf'))
        attn_weights = torch.softmax(QK_scaled, dim=-1)  # shape: (batch_size, seq_len, seq_len)
        attn_output = torch.bmm(attn_weights, V)  # shape: (batch_size, seq_len, V_dim)
        return attn_output.to(self.device)

        """

        d_k = Q.size(-1)
        # Compute dot product of query and key for each head
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # Apply mask
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        # Apply softmax activation function
        attn_weights = torch.softmax(attn_logits, dim=-1)
        # Multiply weights by values
        output = torch.matmul(attn_weights, V)

        return output, attn_weights
        """


    def forward(self, query, key, value, mask):
        '''
        Multiply query, key and value with their respectively projection matrices to obtain Q, K and V
        before calculating scaled dot-product attention. 
        '''
        Q = self.query_matrix(query)
        K = self.key_matrix(key)
        V = self.value_matrix(value)
        attention = self.scaled_dot_product_attention(Q, K, V, mask)

        return attention



'''
Multi-head attention, which linearly project the queries, keys and values 'num_heads' times 
with different learned linear projections.
'''
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, model_dim, Q_dim, K_dim, V_dim, device):
        super().__init__()

        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList(
            [Attention(model_dim, Q_dim, K_dim, V_dim, device) for _ in range(self.num_heads)]
        )
        self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim)
        self.model_dim = model_dim
        self.device = device
    

    ## Q2(1 points): implement multi-head attention as follows: 
    ## MultiHead(query, key, value, mask) = Concat(head_1, ..., head_h)W, 
    # where head_i = Attention(query, key, value, mask) and W is a projection matrix.
    def forward(self, query, key, value, mask):
        ## TO DO
        # multihead_attention = torch.zeros(query.shape[0], query.shape[1], self.model_dim).to(self.device)

        attention_heads_output = [self.attention_heads[i](query, key, value, mask) for i in range(self.num_heads)]
        concat_attention_heads_output = torch.cat(attention_heads_output, dim=-1)
        multihead_attention = self.projection_matrix(concat_attention_heads_output)

        return multihead_attention



'''
Positional encoding to make use of the sequence order information. 
'''
class PositionalEncoding(nn.Module):

    ## Q3(3 points): implement positional encoding as follows:
    ## PE(pos,2i) = sin(pos/(10000^(2i/model_dim)), PE(pos,2i+1) = cos(pos/(10000^(2i/model_dim))
    ## where pos is the position and i is the dimension.
    def __init__(self, max_len, model_dim, dropout_rate, device):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        ## TO DO
        ## Hint: tensor slicing and torch.arange() might be useful here. 
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        self.pos_encoding = torch.zeros((1, max_len, model_dim)).to(device)
        self.pos_encoding[0, :, 0::2] = torch.sin(pos * div)
        self.pos_encoding[0, :, 1::2] = torch.cos(pos * div)

        print("MAX LENGTH and model dim")
        print(max_len)
        print(model_dim)
        # self.register_buffer('pos_encoding', self.pos_encoding)
    

    def forward(self, x):
        '''
        Add positional information to x. 
        x.shape: (batch_size, seq_len, model_dim)
        '''
        seq_len = x.shape[1]
        # pos_info = self.pos_encoding.unsqueeze(0) # pos_info.shape: (1, max_len, model_dim)
        pos_info = self.pos_encoding  # pos_info.shape: (1, max_len, model_dim)
        # print("POS INFO shape", pos_info.size())

        return self.dropout(x + pos_info[:, :seq_len, :].to(x.device))



'''
Position-wise feed-forward network, which consists of 
two linear transformations with a ReLU in between.
'''
class PositionwiseFeedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.linear_W1 = nn.Linear(model_dim, hidden_dim)
        self.linear_W2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    

    def forward(self, x):
        return self.dropout(self.linear_W2(self.relu(self.linear_W1(x))))



'''
Add & Norm: residual connection and layer normalization. 
'''
class AddAndNorm(nn.Module):

    def __init__(self, model_dim, dropout_rate, device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device

    
    ## Q4(1 point): implement Add & Norm layer using: layer_norm(x + dropout(sublayer(x))),
    ## where sublayer is the function implemented by the sub-layer itself. 
    def forward(self, x, sublayer):
        ## TO DO
        # output = torch.zeros(x.shape).to(self.device)
        output = self.layer_norm(x + self.dropout(sublayer(x))).to(self.device)

        return output



'''
Single encoder layer of transformer, which consists of a multi-head attention layer,
two add & norm layers and a position-wise feed-forward layer. 
'''
class EncoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads, hidden_dim, dropout_rate, device):
        super().__init__()

        Q_dim = K_dim = V_dim = model_dim // num_heads
        self.multihead_attn = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim, device)
        self.add_norm_1 = AddAndNorm(model_dim, dropout_rate, device) 
        self.pos_feedforward = PositionwiseFeedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_2 = AddAndNorm(model_dim, dropout_rate, device)
    

    def forward(self, x):
        add_norm_1_output = self.add_norm_1(x, lambda x: self.multihead_attn(x, x, x, mask = False))
        add_norm_2_output = self.add_norm_2(add_norm_1_output, self.pos_feedforward)

        return add_norm_2_output



'''
Encoder of transformer, which consists of 'num_layers' of encoder layers. 
'''
class Encoder(nn.Module):

    def __init__(self, src_vocab_size, num_layers, model_dim, num_heads, hidden_dim, dropout_rate, max_len, device):
        super().__init__()

        self.model_dim = model_dim
        self.src_token_emb = nn.Embedding(src_vocab_size, model_dim) # token embedding
        self.src_pos_emb = PositionalEncoding(max_len, model_dim, dropout_rate, device) # positional embedding
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, hidden_dim, dropout_rate, device) for _ in range(num_layers)])
    

    def forward(self, source):
        # We multiply token embedding by 'self.model_dim**0.5' before positional embedding to make 
        # the positional encoding relatively smaller. This means the original meaning in the embedding 
        # vector won’t be lost when we add them together.
        output = self.src_pos_emb(self.src_token_emb(source.long()) * self.model_dim**0.5) 
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output)
        
        return output



'''
Single decoder layer of transformer, which consists of a masked multi-head attention layer, 
a multi-head attention layer, three add & norm layers and a position-wise feed-forward layer. 
'''
class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads, hidden_dim, dropout_rate, device):
        super().__init__()

        Q_dim = K_dim = V_dim = model_dim // num_heads
        self.masked_multihead_attn = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim, device)
        self.add_norm_1 = AddAndNorm(model_dim, dropout_rate, device) 
        self.enc_dec_multihead_attn = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim, device)
        self.add_norm_2 = AddAndNorm(model_dim, dropout_rate, device) 
        self.pos_feedforward = PositionwiseFeedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_3 = AddAndNorm(model_dim, dropout_rate, device)
    

    def forward(self, x, memory):
        # mask out the leftward information flow in the decoder to preserve the auto-regressive property. 
        add_norm_1_output = self.add_norm_1(x, lambda x: self.masked_multihead_attn(x, x, x, mask = True))
        add_norm_2_output = self.add_norm_2(add_norm_1_output, lambda add_norm_1_output: self.enc_dec_multihead_attn(add_norm_1_output, memory, memory, mask = False))
        add_norm_3_output = self.add_norm_3(add_norm_2_output, self.pos_feedforward)

        return add_norm_3_output



'''
Decoder of transformer, which consists of 'num_layers' of decoder layers. 
'''
class Decoder(nn.Module):

    def __init__(self, tgt_vocab_size, num_layers, model_dim, num_heads, hidden_dim, dropout_rate, max_len, device):
        super().__init__()

        self.model_dim = model_dim
        self.tgt_token_emb = nn.Embedding(tgt_vocab_size, model_dim)
        self.tgt_pos_emb = PositionalEncoding(max_len, model_dim, dropout_rate, device)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(model_dim, num_heads, hidden_dim, dropout_rate, device) for _ in range(num_layers)])
        self.linear = nn.Linear(model_dim, tgt_vocab_size)
    

    def forward(self, target, encoder_output):
        output = self.tgt_pos_emb(self.tgt_token_emb(target.long()) * self.model_dim**0.5)
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, encoder_output)
        output = self.linear(output)
        output = F.softmax(output, dim = -1)

        return output



'''
Transformer is made up of 'num_encoder_layers' of encoders and 'num_decoder_layers' of decoders. 
'''
class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, num_encoder_layers, num_decoder_layers, model_dim, num_heads, hidden_dim, dropout_rate, max_len, device):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, num_encoder_layers, model_dim, num_heads, hidden_dim, dropout_rate, max_len, device)
        self.decoder = Decoder(tgt_vocab_size, num_decoder_layers, model_dim, num_heads, hidden_dim, dropout_rate, max_len, device)
    

    def forward(self, source, target):
        output = self.decoder(target, self.encoder(source))

        return output




'''
Test case.
'''
if __name__ == '__main__':
    src_vocab_size = 20
    tgt_vocab_size = 20
    num_encoder_layers = 6
    num_decoder_layers = 6 
    model_dim = 512 
    num_heads = 8
    hidden_dim = 2048
    dropout_rate = 0.1 
    max_len = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = torch.LongTensor([[0,18,3,4,5,10,11,15,2,7,6,4,8,13,1],[0,3,7,11,12,16,14,5,7,8,18,17,9,6,1]]).to(device)
    target = torch.LongTensor([[0,14,2,8,17,10,1,16,11,2,10,13,4,2,1],[0,17,1,16,8,9,7,0,18,7,15,3,9,10,1]]).to(device)
    transformer = Transformer(src_vocab_size, tgt_vocab_size, num_encoder_layers, num_decoder_layers, model_dim, num_heads, hidden_dim, dropout_rate, max_len, device).to(device)
    print(transformer(source, target))




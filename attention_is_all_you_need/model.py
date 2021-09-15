import torch
from torch import nn
from main import *

# d_model = 512
# num_heads = 8
# d_k = d_model // num_heads


class Transformer(nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc = nn.Linear()

    def forward(self, x):
        encoded = self.encoder.encode(x)
        decoded = self.decoder.decode(encoded)
        output = self.fc(decoded)
        return output


class Encoder():
    def __init__(self, vocab_size, d_model, num_layers=6):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.positional_encoding = PositionalEncoding()
        self.residual_connection = ResidualConnection()

        self.layer_normalize = LayerNormalize()
        self.dropout = nn.Dropout(p=0.1)

        self.position_wise_feed_forward =\
            torch.max(0, x * W_1 + b_1) * W_2 + b_2

    def encode(self, x):
        # Embedding
        embeded = self.embedding(x)

        # Positional Encoding
        positional_encoded = embeded + self.positional_encoding(x)

        # multi_head_attention
        multi_head_attention = MultiHeadAttention(
            Q=positional_encoded, K=positional_encoded, V=positional_encoded,
            d_k=self.d_model).multi_head_attention()

        # Residual Connection
        output = self.residual_connection(
            multi_head_attention, positional_encoded)

        # Normalization
        output = self.layer_normalize(output)

        # Feed Forward
        position_wise_feed_forward = self.position_wise_feed_forward(output)

        # Residual Connection
        output = self.residual_connection(output, position_wise_feed_forward)

        # Normalization
        output = self.layer_normalize(output)

        # return output
        return nn.ReLU(output)


class PositionalEncoding():
    def __init__(self):
        pass

    def positional_encoding(self, i):
        if i % 2 == 0:
            return sin(pos / 10000 ** (2i / d_model))
        else:
            return cos(pos / 10000 ** (2i / d_model))


class ScaledDotProductAttention():
    def scaled_dot_product_attension(self, q, k, v, d_k):
        # q shape: (batch, input_dim, output_dim)
        # k shape: (batch, input_dim, output_dim)

        # key vector transpose shape: (batch, output_dim, vocab_size)
        return torch.matmul(
            nn.Softmax(
                torch.divide(
                    torch.matmul(q, k.permute(0, 2, 1).contiguous()),
                    (d_k ** 0.5)
                )
            ), v
        )


class MultiHeadAttention():
    def __init__(self, Q, K, V, d_k):
        self.Q = Q
        self.K = K
        self.V = V
        self.d_k = d_k
        self.scaled_dot_product_attension = ScaledDotProductAttention()
        self.Linear = nn.Linear(d_k, d_k)

    def multi_head_attention(self):
        attentions = self.scaled_dot_product_attension.scaled_dot_product_attension(
            self.Q[0], self.K[0], self.V[0], self.d_k)
        for i in range(1, len(num_heads)):
            attentions = torch.cat(attentions,
                                   self.scaled_dot_product_attension.scaled_dot_product_attension(
                                       self.Linear(self.Q[i]), self.Linear(
                                           self.K[i]), self.Linear(self.V[i]),
                                       self.d_k)
                                   )
        output = self.Linear(attentions)
        return output


class Decoder():
    def __init__(self):
        self.positional_encoding = PositionalEncoding()
        self.masked_multi_head_attention = MaskedMultiHeadAttention()
        self.residual_connection = ResidualConnection()

        self.fc = nn.Linear()

        self.position_wise_feed_forward =\
            torch.max(0, x * W_1 + b_1) * W_2 + b_2

    def decode(self, x):

        output = self.fc(x)
        return nn.Softmax(output)


class MaskedMultiHeadAttention():
    pass


class ResidualConnection():
    pass


class LayerNormalize():
    pass

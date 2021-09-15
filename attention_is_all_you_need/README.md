# Attention Is All You Need
Implementation Transformer Architecture

## Purpose

- over 28.4 BLEU score on the Multi30k(torchtext datasets) English-to-German translation task
    - 4.5M sentence encoded using byte-pair-encoding pairs
    - source-target vocab of about 37000 tokens

- [paper_link](https://arxiv.org/abs/1706.03762)

## Architecture Summary

### Encoder
- 6 stack layers
- Positional Encoding
- Multi-Head Attention
- Residual Connection
- Layer Normalization ($LayerNorm(x + Sublayer(x)))
- Feed Forward

### Decoder
- 6 stack layers
- Positional Encoding
- Masked Multi-Head Attention
- Multi-Head Attention
- Residual Connection
- Layer Normalization ($LayerNorm(x + Sublayer(x)))
- Feed Forward
- Linear & Softmax Layer

### Attention
- Query, Key, Value vectors
- Scaled Dot-Product Attention (Attention(Q,\ K,\ V) = softmax({{QK^T \over \sqrt{d_k}}}) V)
- Multi-Head Attention

## Hyper Parameters

### Optimizer
- Adam

### Regularization
- Dropout (p=0.1)
- Labem Smoothing (value=0.1)
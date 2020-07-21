import numpy as np
import torch
from eight_mile.pytorch.layers import SeqScaledDotProductRelativeAttention, SeqScaledWindowedRelativeAttention, sequence_mask


def make_rpr(rpr_key_emb, rpr_value_emb, rpr_k, seq_len):
    seq = torch.arange(seq_len)
    window_len = 2 * rpr_k
    edges = seq.view(1, -1) - seq.view(-1, 1) + rpr_k
    edges = torch.clamp(edges, 0, window_len)
    return rpr_key_emb(edges), rpr_value_emb(edges)


def unfold_rpr(rpr_key_emb, rpr_value_emb, rpr_k):
    window_len = 2 * rpr_k + 1
    window = torch.arange(window_len)
    return rpr_key_emb(window), rpr_value_emb(window)


def test_windowed_ra():
    num_heads = 4
    d_model = 64
    rpr_k = 1
    batchsize = 2
    nctx = 12
    d_k = d_model // num_heads

    old = SeqScaledDotProductRelativeAttention(pdrop=0.)
    new = SeqScaledWindowedRelativeAttention(pdrop=0.)

    rpr_key_emb = torch.nn.Embedding(2 * rpr_k + 1, d_k)
    rpr_value_emb = torch.nn.Embedding(2 * rpr_k + 1, d_k)

    Q = torch.randn(batchsize, num_heads, nctx, d_k)
    K = torch.randn(batchsize, num_heads, nctx, d_k)
    V = torch.randn(batchsize, num_heads, nctx, d_k)
    lengths = torch.randint(2, nctx, [batchsize, ])
    seq_mask = sequence_mask(lengths, max_len=nctx)
    in_mask = seq_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
    out_mask = seq_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]

    # manually create a ra_mask to prevent attention beyond rpr_k
    ones = torch.ones(nctx, nctx)
    ra_mask = torch.triu(ones, diagonal=-rpr_k) - torch.triu(ones, diagonal=rpr_k + 1)
    mask = in_mask * ra_mask.unsqueeze(0).unsqueeze(0)
    rpr_key_old, rpr_value_old = make_rpr(rpr_key_emb, rpr_value_emb, rpr_k, nctx)
    old.eval()
    out_old = old((Q, K, V, rpr_key_old, rpr_value_old, mask))
    out_old = out_old.masked_fill(out_mask == False, 1).detach().numpy()
    print(out_old.shape)

    # using the windowed relative attention with the original sequence mask
    rpr_key_new, rpr_value_new = unfold_rpr(rpr_key_emb, rpr_value_emb, rpr_k)
    new.eval()
    out_new = new((Q, K, V, rpr_key_new, rpr_value_new, in_mask))
    out_new = out_new.masked_fill(out_mask == False, 1).detach().numpy()
    print(out_new.shape)

    assert np.allclose(out_old, out_new, atol=1e-6)

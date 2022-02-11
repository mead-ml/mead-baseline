from collections import namedtuple
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eight_mile.pytorch.layers import (
    SeqDotProductAttentionT5,
    SeqScaledDotProductAttentionT5,
)
NH = 4
NQ = 7
NK = 6
NB = 32

@pytest.fixture
def generate_buckets_values():
    REL_BUCKETS = torch.tensor([[0, 17, 18, 19, 20, 21],
                                [1, 0, 17, 18, 19, 20],
                                [2, 1, 0, 17, 18, 19],
                                [3, 2, 1, 0, 17, 18],
                                [4, 3, 2, 1, 0, 17],
                                [5, 4, 3, 2, 1, 0],
                                [6, 5, 4, 3, 2, 1]])

    REL_EMB = torch.tensor([[[0., 17., 18., 19., 20., 21.],
                             [1., 0., 17., 18., 19., 20.],
                             [2., 1., 0., 17., 18., 19.],
                             [3., 2., 1., 0., 17., 18.],
                             [4., 3., 2., 1., 0., 17.],
                             [5., 4., 3., 2., 1., 0.],
                             [6., 5., 4., 3., 2., 1.]],

                            [[32., 49., 50., 51., 52., 53.],
                             [33., 32., 49., 50., 51., 52.],
                             [34., 33., 32., 49., 50., 51.],
                             [35., 34., 33., 32., 49., 50.],
                             [36., 35., 34., 33., 32., 49.],
                             [37., 36., 35., 34., 33., 32.],
                             [38., 37., 36., 35., 34., 33.]],

                            [[64., 81., 82., 83., 84., 85.],
                             [65., 64., 81., 82., 83., 84.],
                             [66., 65., 64., 81., 82., 83.],
                             [67., 66., 65., 64., 81., 82.],
                             [68., 67., 66., 65., 64., 81.],
                             [69., 68., 67., 66., 65., 64.],
                             [70., 69., 68., 67., 66., 65.]],

                            [[96., 113., 114., 115., 116., 117.],
                             [97., 96., 113., 114., 115., 116.],
                             [98., 97., 96., 113., 114., 115.],
                             [99., 98., 97., 96., 113., 114.],
                             [100., 99., 98., 97., 96., 113.],
                             [101., 100., 99., 98., 97., 96.],
                             [102., 101., 100., 99., 98., 97.]]])

    return REL_BUCKETS, REL_EMB



def test_rel_buckets_dp(generate_buckets_values):
    buckets, rel_emb = generate_buckets_values
    dp = SeqDotProductAttentionT5(0, NH)
    dp.rel_embedding = torch.nn.Parameter(torch.tensor(np.arange((NH*NB), dtype=np.float32).reshape(NH, NB)))
    memory_position = torch.arange(NK).view(1, -1)
    query_position = torch.arange(NQ).view(-1, 1)
    relative_position = memory_position - query_position
    rp_bucket = dp._relative_position_bucket(relative_position)
    np.allclose(rp_bucket.numpy(), buckets.numpy())
    np.allclose(rel_emb.numpy(), dp.rel_embedding[:, rp_bucket].detach().numpy())


def test_rel_buckets_sdp(generate_buckets_values):
    buckets, rel_emb = generate_buckets_values
    sdp = SeqScaledDotProductAttentionT5(0, NH)
    sdp.rel_embedding = torch.nn.Parameter(torch.tensor(np.arange((NH*NB), dtype=np.float32).reshape(NH, NB)))
    memory_position = torch.arange(NK).view(1, -1)
    query_position = torch.arange(NQ).view(-1, 1)
    relative_position = memory_position - query_position
    rp_bucket = sdp._relative_position_bucket(relative_position)
    np.allclose(rp_bucket.numpy(), buckets.numpy())
    np.allclose(rel_emb.numpy(), sdp.rel_embedding[:, rp_bucket].detach().numpy())

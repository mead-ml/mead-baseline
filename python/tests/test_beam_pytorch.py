import os
import math
import json
import pytest
import numpy as np
from mock import patch, MagicMock
torch = pytest.importorskip('torch')
from baseline.utils import Offsets
from baseline.pytorch.torchy import vec_log_sum_exp
from eight_mile.pytorch.layers import BeamSearchBase, repeat_batch

B = 1
V = 3  # It has to be at least 3 for Offsets.EOS
K = 2

PATH_1 = [[0.7, 0.2, 0.1], [0.4, 0.3, 0.3], [0.6, 0.3, 0.1], [0.1, 0.1, 0.9]]
BEST_1_1ST = [0, 0, 0, 2]
BEST_1_2ND = [0, 1, 0, 2]

PATH_2 = [[0.1, 0.2, 0.7], [0.7, 0.2, 0.1], [0.7, 0.2, 0.1], [0.2, 0.2, 0.6]]
BEST_2_1ST = [[2, 2, 2, 2]]
BEST_2_2ND = [[1, 0, 0, 2]]


class MockBeamSearch(BeamSearchBase):

    def __init__(self, path_dist):
        super().__init__(beam=K)
        self.path_dist = path_dist
        self.i = 0

    def init(self, encoder_outputs):
        """Tile batches for encoder inputs and the likes."""

    def step(self, paths, _):

        probs = np.array(self.path_dist[self.i], dtype=np.float32)
        self.i += 1
        probs = probs.reshape((B, V))
        single = torch.from_numpy(probs).log()
        return repeat_batch(single, self.K), None

    def update(self, beams, _):
        """Select the correct hidden states and outputs to used based on the best performing beams."""
        return None

def test_beam_easy():
    from collections import namedtuple
    # Always pick the right path
    encoder = namedtuple("EncoderOutput", "output src_mask")
    probs = np.zeros((B, V), dtype=np.float32)
    probs[0, 1] = 0.33
    probs[0, 0] = 0.33
    probs[0, 2] = 0.33
    encoder.output = torch.from_numpy(probs).log()
    mbs = MockBeamSearch(PATH_1)
    paths, lengths, probs = mbs(encoder)
    paths = paths.squeeze()
    assert np.allclose(paths[0].numpy(), np.array(BEST_1_1ST))
    assert np.allclose(paths[1].numpy(), np.array(BEST_1_2ND))


def test_beam_lengths():
    from collections import namedtuple
    # Always pick the right path
    encoder = namedtuple("EncoderOutput", "output src_mask")
    probs = np.zeros((B, V), dtype=np.float32)
    probs[0, 1] = 0.33
    probs[0, 0] = 0.33
    probs[0, 2] = 0.33
    encoder.output = torch.from_numpy(probs).log()
    mbs = MockBeamSearch(PATH_2)
    paths, lengths, probs = mbs(encoder)
    paths = paths.squeeze()
    assert np.allclose(paths[0].numpy(), np.array(BEST_2_1ST))
    assert np.allclose(paths[1].numpy(), np.array(BEST_2_2ND))

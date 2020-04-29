import os
from collections import namedtuple
import pytest
import numpy as np
from eight_mile.utils import get_version

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.skipif(get_version(tf) < 2, reason="TF1.X")
from eight_mile.tf.layers import BeamSearchBase, repeat_batch

B = 1
V = 3  # It has to be at least 3 for Offsets.EOS
K = 2

"""
Originally the probs in path one for index 1 were `[0.4, 0.3, 0.3]`

This lead to a mis-match between the tensorflow and pytorch results.
As you can see below when there are multiple values that are the same
in tf then `tf.math.top_k` will select the first one of that value.
When there are duplicates pytorch `.topk` will select the last one.

From the tensorflow docs:
    `If two elements are equal, the lower-index element appears first.`

Pytorch docs don't explain what they do.

This resulted in pytorch yielding the paths

`[0, 2, 2, 2]` and `[0, 0, 0, 2]` which we wrong based on our gold
which was assuming a select the first approach.

(Pdb) flat_scores
<tf.Tensor: id=161, shape=(1, 6), dtype=float32, numpy=
array([[-1.2729657, -1.5606477, -1.5606477, -2.5257287, -2.8134108,
        -2.8134108]], dtype=float32)>
(Pdb) n
> /home/blester/dev/work/baseline/python/eight_mile/tf/layers.py(1988)__call__()
-> probs = tf.reshape(probs, (bsz, -1))
(Pdb) best_idx
<tf.Tensor: id=164, shape=(1, 2), dtype=int32, numpy=array([[0, 1]], dtype=int32)>

(Pdb) flat_scores
tensor([[-1.2730, -1.5606, -1.5606, -2.5257, -2.8134, -2.8134]])
(Pdb) best_idx
tensor([[0, 2]])
"""

PATH_1 = [[0.7, 0.2, 0.1], [0.4, 0.35, 0.25], [0.6, 0.3, 0.1], [0.1, 0.1, 0.9]]
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
        single = np.log(probs)
        single = tf.convert_to_tensor(single)
        return repeat_batch(single, self.K), None

    def update(self, beams, _):
        """Select the correct hidden states and outputs to used based on the best performing beams."""
        return None


def test_beam_easy():
    # Always pick the right path
    encoder = namedtuple("EncoderOutput", "output src_mask")
    probs = np.zeros((B, V), dtype=np.float32)
    probs[0, 1] = 0.33
    probs[0, 0] = 0.33
    probs[0, 2] = 0.33
    encoder.output = np.log(probs)
    mbs = MockBeamSearch(PATH_1)
    paths, lengths, probs = mbs(encoder)
    paths = paths.numpy().squeeze()
    assert np.allclose(paths[0], np.array(BEST_1_1ST))
    assert np.allclose(paths[1], np.array(BEST_1_2ND))


def test_beam_lengths():
    # Always pick the right path
    encoder = namedtuple("EncoderOutput", "output src_mask")
    probs = np.zeros((B, V), dtype=np.float32)
    probs[0, 1] = 0.33
    probs[0, 0] = 0.33
    probs[0, 2] = 0.33
    encoder.output = np.log(probs)
    mbs = MockBeamSearch(PATH_2)
    paths, lengths, probs = mbs(encoder)
    paths = paths.numpy().squeeze()
    assert np.allclose(paths[0], np.array(BEST_2_1ST))
    assert np.allclose(paths[1], np.array(BEST_2_2ND))

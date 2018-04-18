import unittest
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from baseline.pytorch.torchy import sequence_mask

class MaskTest(unittest.TestCase):

    def setUp(self):
        self.batch_size = np.random.randint(5, 10)
        max_seq = np.random.randint(15, 20)
        self.lengths = Variable(torch.LongTensor(np.random.randint(1, max_seq, size=[self.batch_size])))
        self.seq_len = torch.max(self.lengths).data[0]
        self.scores = Variable(torch.rand(self.batch_size, self.seq_len))

    def test_mask_shape(self):
        mask = sequence_mask(self.lengths)
        self.assertEqual(mask.size(0), self.batch_size)
        self.assertEqual(mask.size(1), self.seq_len)

    def test_mask(self):
        mask = sequence_mask(self.lengths)
        np_mask = np.zeros((self.batch_size, self.seq_len))
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                if j < self.lengths.data[i]:
                    np_mask[i, j] = 1
        np.testing.assert_allclose(mask.data.numpy(), np_mask)

    def test_attention_masked_valid_probs(self):
        mask = sequence_mask(self.lengths)
        score_mask = self.scores.masked_fill_(mask, -1e9)
        attention_weights = F.softmax(score_mask, dim=1)
        for row in attention_weights:
            np.testing.assert_allclose(torch.sum(row).data[0], 1)

    def test_attention_masked_ignores_pad(self):
        mask = sequence_mask(self.lengths)
        score_mask = self.scores.masked_fill_(mask, -1e9)
        attention_weights = F.softmax(score_mask, dim=1)
        for row, length in zip(attention_weights, self.lengths):
            if length.data[0] == self.seq_len:
                continue
            masked = row[:length.data[0]]
            np.testing.assert_allclose(masked.data.numpy(), 0.0)

if __name__ == "__main__":
    unittest.main()

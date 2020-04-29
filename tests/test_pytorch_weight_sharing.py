import pytest
import numpy as np

torch = pytest.importorskip("torch")
import torch.nn as nn
from eight_mile.pytorch.layers import pytorch_linear


class TiedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.tgt_embeddings = nn.Embedding(100, 10)  # vsz, dsz
        self.preds = pytorch_linear(10, 100)  # hsz, output_sz
        self.preds.weight = self.tgt_embeddings.weight  # tied weights

    def forward(self, input_vec):
        return self.preds(self.tgt_embeddings(input_vec))


def test_weight_tying():
    model = TiedWeights()
    model.train()

    loss_fn = nn.MSELoss()

    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    steps = 0
    epoch_loss, epoch_accuracy = 0, 0
    for idx in range(100):
        inputs, targets = random_ints(10), random_floats(10)

        likelihood = model(inputs)

        loss = loss_fn(likelihood, targets)

        loss.backward()
        optim.step()

    assert torch.equal(model.tgt_embeddings.weight, model.preds.weight)


def test_weight_tying_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Cuda not available")

    model = TiedWeights().cuda()
    model.train()

    loss_fn = nn.MSELoss()

    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    steps = 0
    epoch_loss, epoch_accuracy = 0, 0
    for idx in range(100):
        inputs, targets = random_ints(10), random_floats(10)
        inputs = inputs.cuda()
        targets = targets.cuda()

        likelihood = model(inputs)

        loss = loss_fn(likelihood, targets)

        loss.backward()
        optim.step()

    assert torch.equal(model.tgt_embeddings.weight, model.preds.weight)


def random_ints(size):
    return torch.tensor(np.random.randint(0, 100, size=size))


def random_floats(size):
    return torch.rand(size, 100)
    # return torch.tensor(np.random.randint(0, 100, size=size))

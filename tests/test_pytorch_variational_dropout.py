import random
import pytest
import numpy as np

torch = pytest.importorskip("torch")
from eight_mile.pytorch.layers import VariationalDropout


@pytest.fixture
def input_():
    SEQ_LEN = random.randint(20, 101)
    BATCH_SIZE = random.randint(5, 21)
    FEAT_SIZE = random.choice([128, 256, 300])
    return torch.randn((SEQ_LEN, BATCH_SIZE, FEAT_SIZE))


@pytest.fixture
def pdrop():
    return random.choice(np.arange(0.2, 0.8, 0.1))


def test_dropout_same_across_seq(input_, pdrop):
    """Test the dropout masks are the same across a sequence."""
    vd = VariationalDropout(pdrop)
    dropped = vd(input_)
    first_mask = (dropped[0, :, :] == 0).detach().numpy()
    for drop in dropped:
        mask = drop == 0
        np.testing.assert_allclose(first_mask, mask.detach().numpy())


def test_dropout_same_across_seq_batch_first(input_, pdrop):
    """Test the dropout masks are the same across a sequence."""
    vd = VariationalDropout(pdrop, batch_first=True)
    dropped = vd(input_)
    first_mask = (dropped[:, 0, :] == 0).detach().numpy()
    for i in range(dropped.shape[1]):
        mask = dropped[:, i] == 0
        np.testing.assert_allclose(first_mask, mask.detach().numpy())


def test_dropout_probs(input_, pdrop):
    """Test that approximately the right number of units are dropped."""
    vd = VariationalDropout(pdrop)
    dropped = vd(input_)
    initial_non_zero = np.count_nonzero(input_.numpy())
    dropped_non_zero = np.count_nonzero(dropped.detach().numpy())
    np.testing.assert_allclose((dropped_non_zero / float(initial_non_zero)), (1 - pdrop), atol=1e-1)


def test_dropout_scaling(input_, pdrop):
    """Test that the dropout layer scales correctly."""
    vd = VariationalDropout(pdrop)
    dropped = vd(input_)
    re_dropped = input_.masked_fill(dropped == 0, 0)
    dropped_np = dropped.detach().numpy()
    re_dropped = re_dropped.numpy()
    np.testing.assert_allclose(re_dropped * (1 / float(pdrop)), dropped_np)


def test_gradient_flow(input_, pdrop):
    """Test that a gradient can flow through the Variational Dropout."""
    input_.requires_grad_()
    vd = VariationalDropout(pdrop)
    dropped = vd(input_)
    output = torch.mean(dropped)
    loss = torch.pow(torch.sub(torch.Tensor([1.0]), output), 2)
    assert input_.grad is None
    loss.backward()
    assert input_.grad is not None

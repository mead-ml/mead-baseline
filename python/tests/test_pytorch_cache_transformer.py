import copy
import pytest
import numpy as np
from mock import MagicMock
torch = pytest.importorskip('torch')
from baseline.pytorch.torchy import sequence_mask
from baseline.pytorch.transformer import subsequent_mask
from baseline.pytorch.transformer import scaled_dot_product_attention as sdpa
from baseline.pytorch.transformer import dot_product_attention as dpa
from baseline.pytorch.transformer import (
    MultiHeadedAttention,
    TransformerDecoder,
    TransformerDecoderStack,
)
from addons.caching_transformer import (
    CachingTransformerDecoder,
    CachingTransformerDecoderStack,
    CachingSrcMultiHeadedAttention,
    CachingSelfMultiHeadedAttention,
)


B = 32
H = 4
D = 128


@pytest.fixture(autouse=True)
def no_grad():
    with torch.no_grad():
        yield


def test_build_single_cache():
    gold = {'src': {}, 'self': {}}
    cache = CachingTransformerDecoder.build_cache()
    assert cache == gold


def test_build_layered_cache():
    layers = np.random.randint(2, 5)
    gold = [{'self': {}, 'src': {}}] * layers
    ctds = CachingTransformerDecoderStack(TransformerDecoderStack(4, 512, 0.1, layers=layers))
    cache = ctds.build_cache()
    assert cache == gold


@pytest.fixture
def seq_data():
    two = torch.rand(B, 2, D)
    thrid = torch.rand(B, 1, D)
    fourth = torch.rand(B, 1, D)
    three = torch.cat([two, thrid], dim=1)
    four = torch.cat([two, thrid, fourth], dim=1)
    return two, three, four, thrid, fourth


@pytest.fixture
def encoder_data():
    T = 25
    encoder_inputs = torch.rand(B, T, D)
    encoder_lengths = torch.randint(1, T + 1, (B,))
    encoder_lengths[torch.randint(0, B, (B//2,))] = T
    encoder_mask = sequence_mask(encoder_lengths).unsqueeze(1).unsqueeze(1)
    return encoder_inputs, encoder_mask


def test_self_attention_cache(seq_data):
    two, three, four, thrid, fourth = seq_data
    mha = MultiHeadedAttention(H, D)
    cmha = CachingSelfMultiHeadedAttention(mha)
    _, cache = cmha(two, subsequent_mask(2), {})
    assert 'key_pre' in cache
    assert 'value_pre' in cache
    assert cache['key_pre'].shape == (B, H, 2, D // H)
    assert cache['value_pre'].shape == (B, H, 2, D // H)
    old_cache = copy.deepcopy(cache)
    _, cache2 = cmha(thrid, subsequent_mask(1), cache)
    assert cache2['key_pre'].shape == (B, H, 3, D // H)
    assert cache2['value_pre'].shape == (B, H, 3, D // H)
    np.testing.assert_allclose(cache2['key_pre'][:, :, :-1, :], old_cache['key_pre'])
    np.testing.assert_allclose(cache2['value_pre'][:, :, :-1, :], old_cache['value_pre'])


def test_src_attention_cache(seq_data, encoder_data):
    two, three, four, thrid, fourth = seq_data
    enc_input, enc_mask = encoder_data
    mha = MultiHeadedAttention(H, D)
    cmha = CachingSrcMultiHeadedAttention(mha)
    _, cache = cmha(two, enc_input, enc_input, enc_mask, {})
    assert 'key_pre' in cache
    assert 'value_pre' in cache
    old_cache = copy.deepcopy(cache)
    _, cache2 = cmha(thrid, enc_input, enc_input, enc_mask, cache)
    np.testing.assert_allclose(cache2['key_pre'], old_cache['key_pre'])
    np.testing.assert_allclose(cache2['value_pre'], old_cache['value_pre'])


def test_src_attention_uses_cache(seq_data, encoder_data):
    two, three, four, thrid, fourth = seq_data
    enc_input, enc_mask = encoder_data
    mha = MultiHeadedAttention(H, D)
    cmha = CachingSrcMultiHeadedAttention(mha)
    _, cache = cmha(two, enc_input, enc_input, enc_mask, {})
    cmha.w_K.__call__ = MagicMock()
    cmha.w_V.__call__ = MagicMock()
    _, cache = cmha(thrid, enc_input, enc_input, enc_mask, cache)
    cmha.w_K.__call__.assert_not_called()
    cmha.w_V.__call__.assert_not_called()


def test_self_attention_values(seq_data):
    two, three, four, thrid, fourth = seq_data
    mha = MultiHeadedAttention(H, D)
    mha.eval()
    cmha = CachingSelfMultiHeadedAttention(mha)

    full_four = mha(four, four, four, subsequent_mask(4))
    full_three, cache = cmha(three, subsequent_mask(3), {})
    add_four, cache = cmha(fourth, subsequent_mask(1), cache)
    np.testing.assert_allclose(full_four, torch.cat([full_three, add_four], dim=1), atol=1e-6)


def test_src_attention_values(seq_data, encoder_data):
    two, three, four, thrid, fourth = seq_data
    enc_input, enc_mask = encoder_data
    mha = MultiHeadedAttention(H, D)
    mha.eval()
    cmha = CachingSrcMultiHeadedAttention(mha)

    full_four = mha(four, enc_input, enc_input, enc_mask)
    full_three, cache = cmha(three, enc_input, enc_input, enc_mask, {})
    add_four, cache = cmha(fourth, enc_input, enc_input, enc_mask, cache)
    np.testing.assert_allclose(full_four, torch.cat([full_three, add_four], dim=1), atol=1e-6)


def test_cache_transformer_decoder_values(seq_data, encoder_data):
    two, three, four, thrid, fourth = seq_data
    enc_input, enc_mask = encoder_data
    td = TransformerDecoder(H, D, 0.0)
    td.eval()
    ctd = CachingTransformerDecoder(td)

    full_four = td(four, enc_input, enc_mask, subsequent_mask(4))
    full_three, cache = ctd(three, enc_input, enc_mask, subsequent_mask(3), ctd.build_cache())
    add_four, cache = ctd(fourth, enc_input, enc_mask, subsequent_mask(1), cache)
    np.testing.assert_allclose(full_four, torch.cat([full_three, add_four], dim=1), atol=1e-5)


def test_cache_transformer_stack_values(seq_data, encoder_data):
    two, three, four, thrid, fourth = seq_data
    enc_input, enc_mask = encoder_data
    tds = TransformerDecoderStack(H, D, 0.0, layers=3)
    tds.eval()
    ctds = CachingTransformerDecoderStack(tds)

    full_four = tds(four, enc_input, enc_mask, subsequent_mask(4))
    full_two, cache = ctds(two, enc_input, enc_mask, subsequent_mask(2), ctds.build_cache())
    add_three, cache = ctds(thrid, enc_input, enc_mask, subsequent_mask(1), cache)
    add_four, cache = ctds(fourth, enc_input, enc_mask, subsequent_mask(1), cache)

    np.testing.assert_allclose(full_four, torch.cat([full_two, add_three, add_four], dim=1), atol=1e-5)

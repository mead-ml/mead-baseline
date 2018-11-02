try:
    import os
    import random
    import pytest
    import unittest
    from mock import patch, MagicMock
    import numpy as np
    import tensorflow as tf
    from baseline.tf.tfy import parallel_conv, char_word_conv_embeddings, highway_conns, skip_conns
except ImportError:
    raise unittest.SkipTest('Failed to import tensorflow')


os.environ['CUDA_VISIBLE_DEVICES'] = ''

def setup_function(function):
    tf.reset_default_graph()


class ParallelConvTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    @classmethod
    def tearDownClass(cls):
        del os.environ['CUDA_VISIBLE_DEVICES']

    def setUp(self):
        tf.reset_default_graph()
        self.batchsz = random.randint(5, 65)
        self.seqsz = random.randint(5, 11)
        self.embedsz = random.randint(100, 301)
        self.num_filt = random.randint(2, 6)
        self.filtsz = set()
        while len(self.filtsz) != self.num_filt:
            self.filtsz.add(random.randint(2, 7))
        self.filtsz = list(self.filtsz)
        self.motsz = random.randint(128, 257)
        self.nfeat_factor = random.randint(1, 4)
        self.max_feat = random.randint(100, 301)
        self.input = np.random.uniform(size=(self.batchsz, self.seqsz, self.embedsz)).astype(np.float32)
        self.p = tf.placeholder(tf.float32, shape=(None, self.seqsz, self.embedsz))

    def test_output_batch_shape_int_arg(self):
        conv, _ = parallel_conv(self.p, self.filtsz, self.embedsz, self.motsz)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(conv.eval({self.p: self.input}).shape[0], self.batchsz)

    def test_output_batch_shape_list_arg(self):
        motsz = [self.motsz] * len(self.filtsz)
        conv, _ = parallel_conv(self.p, self.filtsz, self.embedsz, motsz)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(conv.eval({self.p: self.input}).shape[0], self.batchsz)

    def test_output_feature_shape_int_arg(self):
        conv, _ = parallel_conv(self.p, self.filtsz, self.embedsz, self.motsz)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(conv.eval({self.p: self.input}).shape[1], self.motsz * self.num_filt)

    def test_output_feature_shape_list_arg(self):
        motsz = [self.nfeat_factor * fsz for fsz in self.filtsz]
        conv, _ = parallel_conv(self.p, self.filtsz, self.embedsz, motsz)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(conv.eval({self.p: self.input}).shape[1], sum(motsz))

    def test_shape_available_int(self):
        """The previous tests test the shape of the actual output tensor. This
        tests the output shape information available when building the graph.
        When tf.squeeze is used on a tensor with a None dimension all shape info
        is lost because it is unknown if the None dimension is 1 and should be
        squeezed out. This cased error later because layers needed shape information
        to set the right size. This test makes sure that needed size information
        is present. This test would have caught the break in the classify method.
        """
        conv, _ = parallel_conv(self.p, self.filtsz, self.embedsz, self.motsz)
        conv_shape = conv.get_shape().as_list()
        self.assertEqual(conv_shape, [None, self.motsz * len(self.filtsz)])

    def test_shape_available_list(self):
        """Same as the `test_shape_available_int` commnet."""
        motsz = [self.nfeat_factor * fsz for fsz in self.filtsz]
        conv, _ = parallel_conv(self.p, self.filtsz, self.embedsz, motsz)
        conv_shape = conv.get_shape().as_list()
        self.assertEqual(conv_shape, [None, sum(motsz)])

    def test_conv_called(self):
        with patch('baseline.tf.tfy.tf.nn.conv2d') as conv_mock:
            conv_mock.return_value = tf.zeros((self.batchsz, 1, self.seqsz, self.motsz))
            conv, _ = parallel_conv(self.p, self.filtsz, self.embedsz, self.motsz)
            self.assertEqual(conv_mock.call_count, self.num_filt)

    def test_list_and_number_args_equal(self):
        with tf.variable_scope("TEST"):
            conv1, _ = parallel_conv(self.p, self.filtsz, self.embedsz, self.motsz)
        with tf.variable_scope("TEST", reuse=True):
            conv2, _ = parallel_conv(self.p, self.filtsz, self.embedsz, [self.motsz] * len(self.filtsz))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            np.testing.assert_allclose(conv1.eval({self.p: self.input}), conv2.eval({self.p: self.input}))

    #@patch('baseline.tf.tfy.parallel_conv')
    #@patch('baseline.tf.tfy.skip_conns')
    #def test_char_word_call_correct(self, skip_mock, conv_mock):
    #    conv_ret = MagicMock()
    #    conv_mock.return_value = conv_ret
    #    _, _ = char_word_conv_embeddings(self.p, self.filtsz, self.embedsz, self.motsz)
    #    conv_mock.assert_called_once_with(self.p, self.filtsz, self.embedsz, self.motsz, tf.nn.tanh)
    #    skip_mock.assert_called_once_with(conv_ret, self.motsz * len(self.filtsz), 1)

    #@patch('baseline.tf.tfy.parallel_conv')
    #@patch('baseline.tf.tfy.highway_conns')
    #def test_char_word_var_fm_call_correct(self, skip_mock, conv_mock):
    #    conv_ret = MagicMock()
    #    conv_mock.return_value = conv_ret
    #    nfeats = [min(self.nfeat_factor * fsz, self.max_feat) for fsz in self.filtsz]
    #    _, _ = char_word_conv_embeddings(self.p, self.filtsz, self.embedsz, self.nfeat_factor, gating=highway_conns)
    #    conv_mock.assert_called_once_with(self.p, self.filtsz, self.embedsz, nfeats, tf.nn.tanh)
    #    skip_mock.assert_called_once_with(conv_ret, sum(nfeats), 2)


def test_dilation():
    # Force T to be even
    T = 1
    while T % 2 == 1: T = np.random.randint(6, 20)
    B = 1
    H = np.random.randint(1, 100)
    # Force D to be even and smaller than T
    D = 1
    while D % 2 == 1: D = np.random.randint(2, T)
    # Create an input that is alternating ones and zeros
    input_ = np.zeros((B, T, H)).astype(np.float32)
    input_[:, ::2, :] = 1.0
    tf_in = tf.constant(input_)
    # Force bias to zeros and weights to one
    def cust_init(getter, *args, **kwargs):
        if len(kwargs['shape']) == 1:
            return tf.zeros(kwargs['shape'])
        return tf.ones(kwargs['shape'])
    with tf.variable_scope('normal', custom_getter=cust_init):
        normal, _ = parallel_conv(tf_in, [2], H, 1)
    with tf.variable_scope('dilation_even', custom_getter=cust_init):
        dilated, _ = parallel_conv(tf_in, [2], H, 1, dilations=D)
    with tf.variable_scope('dilation_odd', custom_getter=cust_init):
        dilated_odd, _ = parallel_conv(tf_in, [2], H, 1, dilations=D + 1)
    with tf.Session() as sess:
        norm = sess.run(normal)
        dil = sess.run(dilated)
        dil_odd = sess.run(dilated_odd)
    # A normal 2d conv will hit one row of zeros and one row of ones
    assert norm[0][0] == H
    # A dilation of that skips an even number of spots will hit either all ones or all zeros
    assert dil[0][0] == 2 * H
    # A dilations of that skips an odd number of spots will hit ones and zeros
    assert dil_odd[0][0] == H


def test_errors_on_wrong_motsz_len():
    H = 300
    in_ = tf.random_normal(shape=[12, 20, H])
    filtsz = [3, 4, 5]
    motsz = [100, 128]
    with pytest.raises(AssertionError):
        out = parallel_conv(in_, filtsz, H, motsz)


def test_errors_on_wrong_dilation_len():
    H = 300
    in_ = tf.random_normal(shape=[12, 20, H])
    filtsz = [3, 4, 5]
    dil = [2, 3, 4, 5]
    with pytest.raises(AssertionError):
        out = parallel_conv(in_, filtsz, H, 128, dilations=dil)


if __name__ == "__main__":
    tf.test.main()

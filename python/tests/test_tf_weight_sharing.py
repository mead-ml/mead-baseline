import numpy as np
import tensorflow as tf
from baseline.tf.tfy import tie_weight

def test_sharing():
    input_ = tf.placeholder(tf.int32, shape=[None])
    weight = tf.get_variable("weight", shape=[100, 200], initializer=tf.random_normal_initializer())
    embed = tf.nn.embedding_lookup(weight, input_)


    tie_shape = [weight.get_shape()[-1], weight.get_shape()[0]]
    with tf.variable_scope("Share", custom_getter=tie_weight(weight, tie_shape)):
        layer = tf.layers.Dense(100, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        out = layer(embed)

    loss = tf.reduce_mean(tf.square(out - 0))
    train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        weights = sess.run([weight, layer.kernel])
        np.testing.assert_allclose(weights[0], weights[1].T)
        print("** Weights Start the Same")
        for _ in range(100):
            sess.run(train_op, {input_: np.random.randint(0, 100, size=10)})

        weights = sess.run([weight, layer.kernel])
        np.testing.assert_allclose(weights[0], weights[1].T)
        print("** Weights Stay the Same")
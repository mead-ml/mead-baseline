import tensorflow as tf
from baseline.model import register_model
from baseline.tf.classify import ClassifierModelBase


@register_model(task='classify', name='res-conv')
class ResidualConvClassifier(ClassifierModelBase):
    """A Conv Net classifier that sums the parallel filters rather than concat."""
    def pool(self, embed, dsz, init, **kwargs):
        filtsz = kwargs['filtsz']
        motsz = kwargs['cmotsz']
        DUMMY_AXIS = 1
        TIME_AXIS = 2
        FEATURE_AXIS = 3
        expanded = tf.expand_dims(embed, DUMMY_AXIS)
        output = tf.zeros((tf.shape(expanded)[0], 1, tf.shape(expanded)[TIME_AXIS], motsz))
        for fsz in filtsz:
            with tf.variable_scope('cmot-%s' % fsz):
                kernel_shape = [1, fsz, dsz, motsz]
                W = tf.get_variable('W', kernel_shape)
                b = tf.get_variable(
                    'b', [motsz],
                    initializer=tf.constant_initializer(0.0)
                )
                conv = tf.nn.conv2d(
                    expanded, W,
                    strides=[1, 1, 1, 1],
                    padding="SAME", name="CONV"
                )
                activation = tf.nn.relu(tf.nn.bias_add(conv, b), 'activation')
                output += activation
        combine = tf.reshape(tf.reduce_max(output, axis=TIME_AXIS), (-1, motsz))
        return combine

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        from baseline.utils import color, Colors
        print(color("="*80, Colors.RED))
        print(color("THIS IS TO SHOW THAT THIS SUBCLASS CREATE IS CALLED WHEN LOADING!", Colors.RED))
        print(color("="*80, Colors.RED))
        return super(ResidualConvClassifier, cls).create(embeddings, labels, **kwargs)

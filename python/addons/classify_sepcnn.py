from baseline.keras.classify import SequentialWordClassifierBase

from keras.layers import (MaxPooling1D,
                          Dropout,
                          SeparableConv1D,
                          GlobalAveragePooling1D)


class SepConvWordClassifier(SequentialWordClassifierBase):
    """
    Model defined in https://developers.google.com/machine-learning/guides/text-classification/step-4
    """
    def __init__(self):
        """
        This model does separable convolution in blocks with max pooling, followed by an average pooling over time
        """
        super(SepConvWordClassifier, self).__init__()

    def _pool(self, dsz, **kwargs):
        """Override the base method from parent to provide text pooling facility.
        Here that is a stack of depthwise separable convolutions with interleaved max pooling followed by a global
        avg-over-time pooling

        :param dsz: Word embeddings dim size
        :param kwargs:
        :return: nothing
        """
        filtsz = kwargs['filtsz']
        blocks = kwargs.get('layers', 2)
        pdrop = kwargs.get('dropout', 0.5)
        cmotsz = kwargs['cmotsz']
        poolsz = kwargs.get('poolsz', 2)

        for _ in range(blocks - 1):
            self.impl.add(Dropout(rate=pdrop))
            self.impl.add(SeparableConv1D(filters=cmotsz, kernel_size=filtsz, activation='relu',
                                          bias_initializer='random_uniform', depthwise_initializer='random_uniform',
                                          padding='same'))
            self.impl.add(SeparableConv1D(filters=cmotsz, kernel_size=filtsz, activation='relu',
                                          bias_initializer='random_uniform', depthwise_initializer='random_uniform',
                                          padding='same'))
            self.impl.add(MaxPooling1D(pool_size=poolsz))

        self.impl.add(SeparableConv1D(filters=cmotsz*2, kernel_size=filtsz, activation='relu',
                                      bias_initializer='random_uniform', depthwise_initializer='random_uniform',
                                      padding='same'))
        self.impl.add(GlobalAveragePooling1D())
        self.impl.add(Dropout(rate=pdrop))


def create_model(embeddings, labels, **kwargs):
    return SepConvWordClassifier.create(embeddings, labels, **kwargs)


def load_model(name, **kwargs):
    return SepConvWordClassifier.load(name, **kwargs)

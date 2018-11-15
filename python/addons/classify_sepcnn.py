from baseline.keras.classify import GraphWordClassifierBase
from baseline.model import register_model
from keras.layers import (MaxPooling1D,
                          Dropout,
                          SeparableConv1D,
                          GlobalAveragePooling1D)

@register_model(task='classify', name='sepcnn')
class SepConvWordClassifier(GraphWordClassifierBase):
    """
    Model defined in https://developers.google.com/machine-learning/guides/text-classification/step-4
    """
    def __init__(self):
        """
        This model does separable convolution in blocks with max pooling, followed by an average pooling over time
        """
        super(SepConvWordClassifier, self).__init__()

    def _pool(self, embed, insz, **kwargs):

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
        

        input_ = embed
        for _ in range(blocks - 1):
            drop1 = Dropout(rate=pdrop)(input_)
            sep1 = SeparableConv1D(filters=cmotsz, kernel_size=filtsz, activation='relu',
                                          bias_initializer='random_uniform', depthwise_initializer='random_uniform',
                                          padding='same')(drop1)
            sep2 = SeparableConv1D(filters=cmotsz, kernel_size=filtsz, activation='relu',
                                          bias_initializer='random_uniform', depthwise_initializer='random_uniform',
                                          padding='same')(sep1)
            input_ = MaxPooling1D(pool_size=poolsz)(sep2)

        sep3 = SeparableConv1D(filters=cmotsz*2, kernel_size=filtsz, activation='relu',
                                      bias_initializer='random_uniform', depthwise_initializer='random_uniform',
                                      padding='same')(input_)
        global_average_pooling = GlobalAveragePooling1D()(sep3)
        drop2 = Dropout(rate=pdrop)(global_average_pooling)
            
        return drop2


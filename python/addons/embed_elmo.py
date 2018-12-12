import os
import tensorflow as tf
import tensorflow_hub as hub
from baseline.utils import write_json
from baseline.embeddings import register_embeddings
from baseline.tf.embeddings import TensorFlowEmbeddings


@register_embeddings(name='elmo')
class ElmoEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.string, [None, None], name=name)

    def __init__(self, name, **kwargs):
        super(ElmoEmbeddings, self).__init__()
        self.vsz = None
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.cache_dir = kwargs.get('cache_dir')
        if self.cache_dir is not None:
            os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser(self.cache_dir)
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.finetune)

    def encode(self, x=None):
        if x is None:
            x = ElmoEmbeddings.create_placeholder(self.name)
        self.x = x
        empty_count = tf.reduce_sum(tf.cast(tf.equal(self.x, ''), tf.int32), axis=1)
        T = tf.shape(self.x)[1]
        lengths = T - empty_count
        return self.elmo(
            inputs={
                'tokens': self.x,
                'sequence_len': lengths,
            },
            signature="tokens", as_dict=True)['elmo']

    def detached_ref(self):
        return ElmoEmbeddings(
            self.name, dsz=self.dsz, vsz=self.vsz, finetune=self.finetune, cache_dir=self.cache_dir
        )

    def save_md(self, target):
        write_json({'vsz': self.vsz, 'dsz': self.dsz}, target)

import tensorflow as tf
from baseline.utils import get_version
from baseline.model import register_model
from baseline.tf.tagger.model import RNNTaggerModel


@register_model(task='tagger', name='with-scores')
class RNNTaggerWithScores(RNNTaggerModel):

    def predict_with_scores(self, batch_dict):
        lengths = batch_dict[self.lengths_key]
        batch_dict = self.make_input(batch_dict)
        if get_version(tf) < 2:
            assert self.impl.path_scores is not None, "Sequence scores are not being calculated, you need to use a CRF or constrained decoding to do this."
            return self.sess.run([self.best, self.impl.path_scores], feed_dict=batch_dict)
        bestv = self(batch_dict).numpy()
        assert self.impl.path_scores is not None, "Sequence scores are not being calculated, you need to use a CRF or constrained decoding to do this."
        scores = self.impl.path_scores
        return bestv, scores.numpy()

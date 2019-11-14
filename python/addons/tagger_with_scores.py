import tensorflow as tf
from baseline.utils import get_version
from baseline.model import register_model
from baseline.tf.tagger.model import RNNTaggerModel
from baseline.services import TaggerService


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


class TaggerWithScoresService(TaggerService):

    def predict(self, tokens, **kwargs):
        export_mapping = kwargs.get('export_mapping', {})  # if empty dict argument was passed
        if not export_mapping:
            export_mapping = {'tokens': 'text'}
        label_field = kwargs.get('label', 'label')
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        examples = self.vectorize(tokens_batch)
        outcomes = self.model.predict_with_scores(examples)
        return self.format_output(outcomes, tokens_batch=tokens_batch, label_field=label_field)
    def format_output(self, predicted, **kwargs):
        predicted, scores = predicted
        paths = super().format_output(predicted, **kwargs)
        return paths, scores

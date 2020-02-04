import os
from baseline.services import TaggerService
from baseline.utils import unzip_files, find_model_basename, read_json, write_json, normalize_backend


class TaggerWithScoresService(TaggerService):
    @classmethod
    def load(cls, bundle, **kwargs):
        be = normalize_backend(kwargs.get('backend', 'tf'))
        if be == 'tf':
           bundle = bundle if os.path.isdir(bundle) else unzip_files(bundle)
           model_basename = find_model_basename(bundle)
           state = read_json(f"{model_basename}.state")
           state['type'] = 'with-scores'
           state['module'] = 'addons.tagger_with_scores'
           state['class'] = 'RNNTaggerWithScores'
           write_json(state, f"{model_basename}.state")
        return super().load(bundle)


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

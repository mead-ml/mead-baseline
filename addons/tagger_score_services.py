#!/usr/bin/env python3


import torch.nn.functional as F
from baseline.services import TaggerService
from eight_mile.pytorch.layers import unsort_batch, sequence_mask, MASK_FALSE


class TaggerSequenceScoreService(TaggerService):
    def predict(self, tokens, **kwargs):
        """
        Utility function to convert lists of sentence tokens to integer value one-hots which
        are then passed to the tagger.  The resultant output is then converted back to label and token
        to be printed.

        This method is not aware of any input features other than words and characters (and lengths).  If you
        wish to use other features and have a custom model that is aware of those, use `predict` directly.

        :param tokens: (``list``) A list of tokens

        """
        preproc = kwargs.get('preproc', None)
        if preproc is not None:
            logger.warning("Warning: Passing `preproc` to `TaggerService.predict` is deprecated.")
        valid_labels_only = kwargs.get('valid_labels_only', True)
        export_mapping = kwargs.get('export_mapping', {})  # if empty dict argument was passed
        if not export_mapping:
            export_mapping = {'tokens': 'text'}
        label_field = kwargs.get('label', 'label')
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        # TODO: here we allow vectorizers even for preproc=server to get `word_lengths`.
        # vectorizers should not be available when preproc=server.
        examples = self.vectorize(tokens_batch)
        if self.preproc == 'server':
            unfeaturized_examples = {}
            for exporter_field in export_mapping:
                unfeaturized_examples[exporter_field] = np.array([" ".join([y[export_mapping[exporter_field]]
                                                                   for y in x]) for x in tokens_batch])
            unfeaturized_examples[self.model.lengths_key] = examples[self.model.lengths_key]  # remote model
            examples = unfeaturized_examples

        self.model.eval()
        inputs, perm_idx = self.model.make_input(examples, perm=True, numpy_to_tensor=True)
        transduced = self.model.transduce(inputs)
        paths, scores = self.model.decoder.decode(transduced, inputs['lengths'])
        norm = self.model.decoder.partition(transduced, inputs['lengths'])
        scores = scores / norm

        paths = unsort_batch(paths, perm_idx)
        scores = unsort_batch(scores, perm_idx)

        return self.format_output(paths, tokens_batch=tokens_batch, label_field=label_field,
                                  vectorized_examples=examples, valid_labels_only=valid_labels_only), scores


class TaggerPosteriorDistributionScoreService(TaggerService):
    def predict(self, tokens, **kwargs):
        """
        Utility function to convert lists of sentence tokens to integer value one-hots which
        are then passed to the tagger.  The resultant output is then converted back to label and token
        to be printed.

        This method is not aware of any input features other than words and characters (and lengths).  If you
        wish to use other features and have a custom model that is aware of those, use `predict` directly.

        :param tokens: (``list``) A list of tokens

        """
        preproc = kwargs.get('preproc', None)
        if preproc is not None:
            logger.warning("Warning: Passing `preproc` to `TaggerService.predict` is deprecated.")
        valid_labels_only = kwargs.get('valid_labels_only', True)

        export_mapping = kwargs.get('export_mapping', {})  # if empty dict argument was passed
        if not export_mapping:
            export_mapping = {'tokens': 'text'}
        label_field = kwargs.get('label', 'label')
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        # TODO: here we allow vectorizers even for preproc=server to get `word_lengths`.
        # vectorizers should not be available when preproc=server.
        examples = self.vectorize(tokens_batch)
        if self.preproc == 'server':
            unfeaturized_examples = {}
            for exporter_field in export_mapping:
                unfeaturized_examples[exporter_field] = np.array([" ".join([y[export_mapping[exporter_field]]
                                                                   for y in x]) for x in tokens_batch])
            unfeaturized_examples[self.model.lengths_key] = examples[self.model.lengths_key]  # remote model
            examples = unfeaturized_examples

        self.model.eval()
        inputs, perm_idx = self.model.make_input(examples, perm=True, numpy_to_tensor=True)
        transduced = self.model.transduce(inputs)
        paths, _ = self.model.decoder.decode(transduced, inputs['lengths'])
        scores = self.model.decoder.posterior(transduced, inputs['lengths'])

        paths = unsort_batch(paths, perm_idx)
        scores = unsort_batch(scores, perm_idx)
        return self.format_output(paths, tokens_batch=tokens_batch, label_field=label_field,
                                  vectorized_examples=examples, valid_labels_only=valid_labels_only), scores


class TaggerTransducedDistributionScoreService(TaggerService):
    def predict(self, tokens, **kwargs):
        """
        Utility function to convert lists of sentence tokens to integer value one-hots which
        are then passed to the tagger.  The resultant output is then converted back to label and token
        to be printed.

        This method is not aware of any input features other than words and characters (and lengths).  If you
        wish to use other features and have a custom model that is aware of those, use `predict` directly.

        :param tokens: (``list``) A list of tokens

        """
        preproc = kwargs.get('preproc', None)
        if preproc is not None:
            logger.warning("Warning: Passing `preproc` to `TaggerService.predict` is deprecated.")
        valid_labels_only = kwargs.get('valid_labels_only', True)

        export_mapping = kwargs.get('export_mapping', {})  # if empty dict argument was passed
        if not export_mapping:
            export_mapping = {'tokens': 'text'}
        label_field = kwargs.get('label', 'label')
        tokens_batch = self.batch_input(tokens)
        self.prepare_vectorizers(tokens_batch)
        # TODO: here we allow vectorizers even for preproc=server to get `word_lengths`.
        # vectorizers should not be available when preproc=server.
        examples = self.vectorize(tokens_batch)
        if self.preproc == 'server':
            unfeaturized_examples = {}
            for exporter_field in export_mapping:
                unfeaturized_examples[exporter_field] = np.array([" ".join([y[export_mapping[exporter_field]]
                                                                   for y in x]) for x in tokens_batch])
            unfeaturized_examples[self.model.lengths_key] = examples[self.model.lengths_key]  # remote model
            examples = unfeaturized_examples

        self.model.eval()
        inputs, perm_idx = self.model.make_input(examples, perm=True, numpy_to_tensor=True)
        transduced = self.model.transduce(inputs)
        paths, _ = self.model.decoder.decode(transduced, inputs['lengths'])

        paths = unsort_batch(paths, perm_idx)
        scores = unsort_batch(transduced, perm_idx)

        scores = F.softmax(scores, dim=-1)
        mask = sequence_mask(inputs['lengths']).to(scores.device).unsqueeze(-1)
        scores = scores.masked_fill(mask == MASK_FALSE, 0.0)
        return self.format_output(paths, tokens_batch=tokens_batch, label_field=label_field,
                                  vectorized_examples=examples, valid_labels_only=valid_labels_only), scores

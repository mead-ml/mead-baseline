import pytest
from baseline import model
from mock import patch, MagicMock
import numpy as np

FULL_CONFIG = {
        "mxlen": 60,
        "mxwlen": 40,
        "maxw": 40,  # add this so this works for both classifiers and taggers
    }

NO_LENGTH_CONFIG = {
        # "mxlen": 60,
        # "mxwlen": 40,
    }

TOKENS = ['this', 'is', 'a', 'test']
CLASS_RESPONSE =  [[('test', 1), ('test', 1)]]
TAG_RESPONSE = [[np.array((1)),np.array((1)),np.array((1)),np.array((1))]]

def create_dummy_classifier(mxlen=None, mxwlen=None, zero_alloc=None):
    """
    fixture to return a dummy classifier.
    """
    m = model.ClassifierModel()
    m.classify = MagicMock(name='classify_method')
    m.classify.return_value = CLASS_RESPONSE

    if mxlen:
        m.mxlen = mxlen
    if mxwlen:
        m.mxwlen = mxwlen
    if zero_alloc:
        m.zero_alloc = zero_alloc

    return m

def create_dummy_tagger(mxlen=None, mxwlen=None, zero_alloc=None):
    """
    fixture to return a dummy tagger.
    """
    m = model.TaggerModel()
    m.predict = MagicMock(name='predict_method')
    m.predict.return_value = TAG_RESPONSE

    m.get_labels = MagicMock(name='get_labels_method')
    m.get_labels.return_value = {'one': 1, 'two': 2, 'three': 3, 'four': 4}

    if mxlen:
        m.mxlen = mxlen
    if mxwlen:
        m.maxw = mxwlen
    if zero_alloc:
        m.zero_alloc = zero_alloc

    return m

@patch('baseline.model.WordCharLength')
def test_classify_config_settings(WordCharLength):
    """
    ensure that the classifier will read from the config in
    classify_text.
    """
    classifier = create_dummy_classifier()

    WordCharLength.predict.return_value = {}

    result = classifier.classify_text(TOKENS, **FULL_CONFIG)

    classifier.classify.assert_called_once()
    WordCharLength.assert_called_with(classifier, 60, 40, np.zeros)

@patch('baseline.model.WordCharLength')
def test_classify_self_settings(WordCharLength):
    """
    ensure that the classifier will read from the classifier in
    classify_text.
    """

    classifier = create_dummy_classifier(mxlen=999, mxwlen=666)

    WordCharLength.predict.return_value = {}

    result = classifier.classify_text(TOKENS, **NO_LENGTH_CONFIG)

    classifier.classify.assert_called_once()
    WordCharLength.assert_called_with(classifier, 999, 666, np.zeros)

@patch('baseline.model.WordCharLength')
def test_classify_token_settings(WordCharLength):
    """
    ensure that the classifier will read from the tokens in
    classify_text.
    """
    # mxlen and mxwlen not set in class. it should be determined from tokens.
    classifier = create_dummy_classifier(mxlen=None, mxwlen=None)

    WordCharLength.predict.return_value = {}

    result = classifier.classify_text(TOKENS, **NO_LENGTH_CONFIG)

    classifier.classify.assert_called_once()
    WordCharLength.assert_called_with(classifier, 4, 4, np.zeros)


@patch('baseline.model.WordCharLength')
def test_tagger_config_settings(WordCharLength):
    """
    ensure that the tagger will read from the config in
    classify_text.
    """
    tagger = create_dummy_tagger()

    WordCharLength.predict.return_value = {'length': [4]}

    result = tagger.predict_text(TOKENS, **FULL_CONFIG)

    tagger.predict.assert_called_once()
    WordCharLength.assert_called_with(tagger, 60, 40, np.zeros)

@patch('baseline.model.WordCharLength')
def test_tagger_self_settings(WordCharLength):
    """
    ensure that the classifier will read from the classifier in
    classify_text.
    """
    tagger = create_dummy_tagger(mxlen=999, mxwlen=666)

    WordCharLength.predict.return_value = {'length': [4]}

    result = tagger.predict_text(TOKENS, **NO_LENGTH_CONFIG)

    tagger.predict.assert_called_once()
    WordCharLength.assert_called_with(tagger, 999, 666, np.zeros)

@patch('baseline.model.WordCharLength')
def test_tagger_token_settings(WordCharLength):
    """
    ensure that the classifier will read from the tokens in
    classify_text.
    """
    # mxlen and mxwlen not set in class. it should be determined from tokens.
    tagger = create_dummy_tagger(mxlen=None, mxwlen=None)

    WordCharLength.predict.return_value = {'length': [4]}

    result = tagger.predict_text(TOKENS, **NO_LENGTH_CONFIG)

    tagger.predict.assert_called_once()
    WordCharLength.assert_called_with(tagger, 4, 4, np.zeros)

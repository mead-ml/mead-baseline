from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
import logging
import numpy as np


def tagger_repl(tagger, **kwargs):
    mxlen = int(kwargs.get('mxlen', 100))
    zeropad = int(kwargs.get('zeropad', 0))
    prompt_name = kwargs.get('prompt', 'class> ')
    history_file = kwargs.get('history_file', '.history')
    history = FileHistory(history_file)

    while True:
        text = prompt(prompt_name, history=history)
        text = text.strip()
        if text == 'quit':
            break
        try:
            tokens = text.split(' ')
            best = tagger.predict_text(mxlen, tokens, zeropad)
            print(best)
        except Exception as e:
            logging.exception('Error')

def classifier_repl(classifier, **kwargs):

    mxlen = int(kwargs.get('mxlen', 100))
    k = int(kwargs.get('k', 1))
    thresh = float(kwargs.get('thresh', 0.0))
    zeropad = int(kwargs.get('zeropad', 0))
    prompt_name = kwargs.get('prompt', 'class> ')
    history_file = kwargs.get('history_file', '.history')
    history = FileHistory(history_file)

    while True:
        text = prompt(prompt_name, history=history)
        text = text.strip()
        if text == 'quit':
            break
        try:
            tokens = text.split(' ')
            outcomes = classifier.classify_text(tokens, mxlen=mxlen, zeropad=zeropad)
            k = min(k, len(outcomes))
            probs = outcomes[:k]
            for prob_i in probs:
                if prob_i[1] > thresh:
                    print('Guess [%s]: %.3f' % prob_i)

        except Exception as e:
            logging.exception('Error')



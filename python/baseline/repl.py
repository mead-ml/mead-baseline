from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
import logging
import numpy as np


def classifier_repl(classifier, **kwargs):

    mxlen = int(kwargs.get('mxlen', 100))
    k = int(kwargs.get('k', 1))
    thresh = float(kwargs.get('thresh', 0.0))
    zeropad = int(kwargs.get('zeropad', 0))
    prompt_name = kwargs.get('prompt', 'class> ')
    history_file = kwargs.get('history_file', '.history')
    history = FileHistory(history_file)
    vocab = classifier.get_vocab()
    halffiltsz = zeropad // 2
    while True:
        unseen_tokens = {}
        text = prompt(prompt_name, history=history)
        text = text.strip()
        if text == 'quit':
            break
        try:
            tokens = text.split(' ')
            x = np.zeros((1, mxlen), dtype=int)
            for j in range(min(len(tokens), mxlen - zeropad + 1)):
                word = tokens[j]
                if word not in vocab:
                    if word != '':
                        print(word)
                        unseen_tokens[word] = 1
                        idx = 0
                else:
                    idx = vocab[word]
                x[0, j + halffiltsz] = idx

            outcomes = classifier.classify(x)[0]
            k = min(k, len(outcomes))
            probs = sorted(outcomes, key=lambda tup: tup[1], reverse=True)[:k]

            for prob_i in probs:
                if prob_i[1] > thresh:
                    print('Guess [%s]: %.3f' % prob_i)

        except Exception as e:
            logging.exception('Error')

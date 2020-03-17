import math
from operator import itemgetter
import numpy as np


def generate_batch(B=None, T=None, H=None):
    B = np.random.randint(5, 11) if B is None else B
    T = np.random.randint(15, 21) if T is None else T
    H = np.random.randint(22, 41) if H is None else H
    scores = np.random.rand(B, T, H).astype(np.float32)
    tags = np.random.randint(1, H, size=(B, T)).astype(np.int64)
    lengths = np.random.randint(1, T, size=(B,)).astype(np.int64)
    lengths[np.random.randint(0, B, size=(B // 2,))] = T
    for s, t, l in zip(scores, tags, lengths):
        s[l:] = 0
        t[l:] = 0
    return scores, tags, lengths


def generate_examples_and_batch(T=None, H=None, diff=None):
    T = np.random.randint(15, 21) if T is None else T
    H = np.random.randint(22, 41) if H is None else H
    diff = np.random.randint(1, T // 2) if diff is None else diff

    item1 = np.random.rand(1, T, H).astype(np.float32)
    tags1 = np.random.randint(1, H, (1, T)).astype(np.int64)

    item2 = np.random.rand(1, T - diff, H).astype(np.float32)
    tags2 = np.random.randint(1, H, (1, T - diff)).astype(np.int64)

    packed_item = np.zeros((2, T, H), dtype=np.float32)
    packed_tags = np.zeros((2, T), dtype=np.int64)
    packed_item[0, :, :] = np.squeeze(item1, 0)
    packed_item[1, :T - diff, :] = np.squeeze(item2, 0)
    packed_tags[0, :] = np.squeeze(tags1, 0)
    packed_tags[1, :T - diff] = np.squeeze(tags2, 0)
    lengths = np.array([T, T - diff])
    return item1, tags1, np.array([T]), item2, tags2, np.array([T - diff]), packed_item, packed_tags, lengths


def explicit_log_sum_exp(xs):
    """Log Sum Exp on a dict of values."""
    max_x = max(xs.values())
    total = 0
    for x in xs.values():
        total += math.exp(x - max_x)
    return max_x + math.log(total)


def explicit_sum(xs):
    return sum(xs.values())


def explicit_score_gold(emiss, trans, golds, start, end):
    score = 0
    for e, g in zip(emiss, golds):
        score += e[g]
    for i in range(len(golds)):
        from_ = start if i == 0 else golds[i - 1]
        to = golds[i]
        score += trans[(from_, to)]
    score += trans[(golds[-1], end)]
    return score


def explicit_forward(emiss, trans, start, end, reduction=explicit_log_sum_exp):
    """Best path through a lattice on the log semiring with explicit looping."""
    trellises = []
    trellis = dict.fromkeys(emiss[0].keys(), -1e4)
    trellis[start] = 0

    for e in emiss:
        new_trellis = {}
        for next_state in trellis:
            score = {}
            for prev_state in trellis:
                score[prev_state] = trellis[prev_state] + e[next_state] + trans[(prev_state, next_state)]
            new_trellis[next_state] = reduction(score)
        trellis = new_trellis
        trellises.append(trellis)
    trellis = {state: trellis[state] + trans[(state, end)] for state in trellis}
    return reduction(trellis), trellises


def explicit_backward(emiss, trans, start, end, reduction=explicit_log_sum_exp):
    new_trans = {(j, i): v for (i, j), v in trans.items()}
    scores, states = explicit_forward(emiss[::-1], new_trans, end, start, reduction)
    return scores, states[::-1]


def explicit_posterior(emiss, trans, start, end, reduction=explicit_log_sum_exp):
    fwd = explicit_forward(emiss, trans, start, end, reduction)[1]
    bwd = explicit_backward(emiss, trans, start, end, reduction)[1]
    joint = []
    for f, b in zip(fwd, bwd):
        joint.append({s: f[s] + b[s] for s in range(len(f))})
    conditional = [{s: j[s] / sum(j.values()) for s in j} for j in joint]
    return conditional


def explicit_posterior_decode(emiss, trans, start, end, reduction=explicit_log_sum_exp):
    posteriors = explicit_posterior(emiss, trans, start, end, reduction)
    scores, tags = [], []
    for posterior in posteriors:
        tag, score = max(posterior.items(), key=itemgetter(1))
        scores.append(score)
        tags.append(tag)
    return tags, sum(scores)


def explicit_trellises_to_dense(trellises):
    return np.stack([explicit_trellis_to_dense(t) for t in trellises])

def explicit_trellis_to_dense(trellis):
    return np.array([trellis[s] for s in range(len(trellis))])


def explicit_nll(emiss, trans, golds, start, end, reduction=explicit_log_sum_exp):
    f = explicit_forward(emiss, trans, start, end, reduction)[0]
    g = explicit_score_gold(emiss, trans, golds, start, end)
    return f - g


def explicit_viterbi(emiss, trans, start, end):
    """Best path through a lattice on the viterbi semiring with explicit looping."""
    backpointers = []
    trellis = dict.fromkeys(emiss[0].keys(), -1e4)
    trellis[start] = 0

    for e in emiss:
        new_trellis = {}
        backpointer = {}
        for next_state in trellis:
            score = {}
            for prev_state in trellis:
                score[prev_state] = trellis[prev_state] + e[next_state] + trans[(prev_state, next_state)]
            new_trellis[next_state] = max(score.values())
            backpointer[next_state] = max(score, key=lambda x: score[x])  # argmax
        trellis = new_trellis
        backpointers.append(backpointer)
    for state in trellis:
        trellis[state] += trans[(state, end)]
    score = max(trellis.values())
    state = max(trellis, key=lambda x: trellis[x])
    states = [state]
    for t in reversed(range(0, len(emiss))):
        states.append(backpointers[t][states[-1]])
    return list(reversed(states[:-1])), score


def build_trans(t):
    """Convert the transition tensor to a dict.

    :param t: `torch.FloatTensor` [H, H]: transition scores in the
        form [to, from]

    :returns: `dict` transition scores in the form [from, to]
    """
    trans = {}
    for i in range(t.shape[0]):
        for j in range(t.shape[0]):
            trans[(i, j)] = t[j, i].item()
    return trans


def build_emission(emission):
    """Convert the emission scores into a list of dicts

    :param emission: `torch.FloatTensor` [T, H]: emission scores

    :returns: `List[dict]`
    """
    es = []
    for emiss in emission:
        e_ = {}
        for i in range(emiss.shape[0]):
            e_[i] = emiss[i].item()
        es.append(e_)
    return es

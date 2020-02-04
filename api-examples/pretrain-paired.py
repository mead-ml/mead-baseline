import logging
import time
import os
from argparse import ArgumentParser
import baseline
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, Dataset
from eight_mile.utils import str2bool, write_json, Offsets, listify
import glob
from baseline.pytorch.embeddings import *
import eight_mile.embeddings
from eight_mile.optz import *
from eight_mile.pytorch.optz import *
from eight_mile.pytorch.layers import TransformerEncoderStack, DenseStack, MultiHeadedAttention
from baseline.vectorizers import Char2DVectorizer, Token1DVectorizer, AbstractVectorizer
from mead.downloader import DataDownloader

import codecs
from collections import Counter
import pandas as pd
from baseline.pytorch.torchy import vec_log_sum_exp
logger = logging.getLogger(__file__)

"""Pre-train a paired model in PyTorch

This file uses Baseline to train a Transformer-based ConveRT
model (https://arxiv.org/pdf/1911.03688.pdf) with PyTorch on multiple GPUs.

If you use `tokens=bpe`, it requires fastBPE.
If you use `tokens=wordpiece` it requires bert_pretrained_pytorch.  
Otherwise, it depends only on six, numpy, pytorch, and baseline.

Because we are trying to pretrain a language model so we can do better on downstream tasks, it probably makes more
sense to train on a full word model, not a model where rare words have already been replaced.

"""
DATASETS = {
    "auto": {
        "train_file": "/data/chats-all/tok/train",
        "valid_file": "/data/chats-all/tok/dev",
        "test_file": "/data/chats-all/tok/test"
    },
    "nav": {
        "train_file": "/data/nav/CHATEXPORT/ab-turns/train",
        "valid_file": "/data/nav/CHATEXPORT/ab-turns/dev",
        "test_file": "/data/nav/CHATEXPORT/ab-turns/test"
    }
}

BERT_TOKENIZER = None

def files_for_dir(directory):
    for file in glob.glob(f'{directory}/*.csv'):
        yield file


class SavableFastBPE(object):
    """Wrapper to define how to pickle fastBPE tokenizer"""
    def __init__(self, codes_path, vocab_path):
        from fastBPE import fastBPE
        self.codes = open(codes_path, 'rb').read()
        self.vocab = open(vocab_path, 'rb').read()
        self.bpe = fastBPE(codes_path, vocab_path)

    def __getstate__(self):
        return {'codes': self.codes, 'vocab': self.vocab}

    def __setstate__(self, state):
        from fastBPE import fastBPE
        import tempfile
        with tempfile.NamedTemporaryFile() as codes, tempfile.NamedTemporaryFile() as vocab:
            codes.write(state['codes'])
            vocab.write(state['vocab'])
            self.bpe = fastBPE(codes.name, vocab.name)

    def apply(self, sentences):
        return self.bpe.apply(sentences)


class BPEVectorizer1D(AbstractVectorizer):
    """Define a Baseline Vectorizer for BPE using fastBPE (https://github.com/glample/fastBPE)

    If you use tokens=bpe, this vectorizer is used, and so then there is a
    dependency on fastBPE

    To use BPE, we assume that a Dictionary of codes and vocab was already created

    """
    def __init__(self, **kwargs):
        """Loads a BPE tokenizer"""
        super().__init__(kwargs.get('transform_fn'))
        self.max_seen = 128
        self.model_file = kwargs.get('model_file')
        self.vocab_file = kwargs.get('vocab_file')
        self.tokenizer = SavableFastBPE(self.model_file, self.vocab_file)
        self.mxlen = kwargs.get('mxlen', -1)
        self.vocab = {k: i for i, k in enumerate(self.read_vocab(self.vocab_file))}

    def read_vocab(self, s):
        vocab = [] + Offsets.VALUES
        with open(s, "r") as f:
            for line in f.readlines():
                token = line.split()[0].strip()
                vocab.append(token)
        return vocab

    def count(self, tokens):
        seen = 0
        counter = Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def iterable(self, tokens):
        for t in tokens:
            if t in Offsets.VALUES:
                yield t
            elif t == '<unk>':
                yield Offsets.VALUES[Offsets.UNK]
            elif t == '<eos>':
                yield Offsets.VALUES[Offsets.EOS]
            else:
                subwords = self.tokenizer.apply([t])[0].split()
                for x in subwords:
                    yield x

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom, 0)  # This shouldnt actually happen
            yield value

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = np.zeros(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


class WordPieceVectorizer1D(AbstractVectorizer):
    """Define a Baseline Vectorizer that can do WordPiece with BERT tokenizer

    If you use tokens=wordpiece, this vectorizer is used, and so then there is
    a dependency on bert_pretrained_pytorch
    """

    def __init__(self, **kwargs):
        """Loads a BertTokenizer using bert_pretrained_pytorch

        :param kwargs:
        """
        super().__init__(kwargs.get('transform_fn'))
        from pytorch_pretrained_bert import BertTokenizer
        self.max_seen = 128
        handle = kwargs.get('embed_file')
        self.tokenizer = BertTokenizer.from_pretrained(handle, do_lower_case=False)
        self.mxlen = kwargs.get('mxlen', -1)

    @property
    def vocab(self):
        return self.tokenizer.vocab

    def count(self, tokens):
        seen = 0
        counter = Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def iterable(self, tokens):
        for tok in tokens:
            if tok == '<unk>':
                yield '[UNK]'
            elif tok == '<EOS>':
                yield '[SEP]'
            else:
                for subtok in self.tokenizer.tokenize(tok):
                    yield subtok

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom)
            if value is None:
                value = vocab['[UNK]']
            yield value

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = np.zeros(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


class TensorDatasetReaderBase(object):
    """Provide a base-class to do operations that are independent of token representation
    """
    def __init__(self, nctx, vectorizer, query_field='content', response_field='response'):
        self.nctx = nctx
        self.vectorizer = vectorizer
        self.query_field = query_field
        self.response_field = response_field
        self.num_words = {}

    def build_vocab(self, directories):
        vocab = Counter()
        for directory in directories:
            logger.info("Reading convos from %s", os.path.basename(directory))
            if directory is None:
                continue

            self.num_words[directory] = 0

            for file in files_for_dir(directory):
                logger.info("Reading turns from %s", os.path.basename(file))
                df = pd.read_csv(file, error_bad_lines=False)
                ##df.fillna('')
                qs = df[self.query_field]
                rs = df[self.response_field]
                logger.info("%d turns", len(qs))
                for q, r in zip(qs, rs):
                    qt = str(q).strip().split()
                    rt = str(r).strip().split()
                    self.num_words[directory] += (len(qt) + len(rt))
                    vocab.update(self.vectorizer.count(qt))
                    vocab.update(self.vectorizer.count(rt))
        return vocab

    def load_features(self, directory, vocabs):

        features = dict()

        x_vector = []
        y_vector = []
        logger.info("Loading features from %s", os.path.basename(directory))
        for file in files_for_dir(directory):
            logger.info("Loading features for convo from %s", os.path.basename(file))
            df = pd.read_csv(file, error_bad_lines=False)
            qs = df[self.query_field]
            ps = df[self.response_field]

            # df.fillna('')

            #x_lengths_vector = []
            #y_lengths_vector = []
            for q, r in zip(qs, ps):
                q = str(q).strip().split()
                r = str(r).strip().split()
                if q == '' or r == '':
                    continue
                q_vec, q_valid_lengths = self.vectorizer.run(q, vocabs)
                x_vector.append(q_vec)
                #x_lengths_vector.append(q_valid_lengths)
                r_vec, r_valid_lengths = self.vectorizer.run(r, vocabs)
                y_vector.append(r_vec)
                #y_lengths_vector.append(r_valid_lengths)
        features['x'] = np.stack(x_vector)
        #features['x_length'] = np.stack(x_lengths_vector)
        features['y'] = np.stack(y_vector)
        #features['y_length'] = np.stack(y_lengths_vector)
        logger.info("x shape: %s", features['x'].shape)
        logger.info("y shape: %s", features['y'].shape)
        return features


class TensorWordDatasetReader(TensorDatasetReaderBase):
    """Read each word, and produce a tensor of x and y that are identical
    """
    def __init__(self, nctx, use_subword=None, model_file=None, vocab_file=None):
        """Create a reader with a context window that reads words

        :param nctx: The context window length
        :param use_subword: If this is not none, it should be either 'bpe' or 'wordpiece'
        """
        self.use_subword = use_subword

        if self.use_subword == 'bpe':
            vectorizer = BPEVectorizer1D(model_file=model_file, vocab_file=vocab_file)
        elif self.use_subword == 'wordpiece':
            vectorizer = WordPieceVectorizer1D(embed_file=model_file)
        else:
            vectorizer = Token1DVectorizer(transform_fn=baseline.lowercase)
        super().__init__(nctx, vectorizer)

    def build_vocab(self, files):
        """Read the vocab file to get the tokens

        :param files:
        :return:
        """
        if self.use_subword is not None:
            super().build_vocab(files)
            return self.vectorizer.vocab
        return super().build_vocab(files)

    def load(self, filename, vocabs):
        features = self.load_features(filename, vocabs)
        x_tensor = torch.tensor(features['x'], dtype=torch.long)
        y_tensor = torch.tensor(features['y'], dtype=torch.long)
        return TensorDataset(x_tensor, y_tensor)

class TensorWordJaggedDatasetReader(TensorWordDatasetReader):
    """Read each word, and produce a tensor of x and y that are only as big as needed, store in a list

    This is done to save memory
    """
    def load_features(self, directory, vocabs):

        features = dict()

        x_vector = []
        y_vector = []
        logger.info("Loading features from %s", os.path.basename(directory))
        for file in files_for_dir(directory):
            logger.info("Loading features for convo from %s", os.path.basename(file))
            df = pd.read_csv(file, error_bad_lines=False)
            qs = df[self.query_field]
            ps = df[self.response_field]

            # df.fillna('')

            #x_lengths_vector = []
            #y_lengths_vector = []
            for q, r in zip(qs, ps):
                q = str(q).strip().split()
                r = str(r).strip().split()
                if q == '' or r == '':
                    continue
                q_vec, q_valid_lengths = self.vectorizer.run(q, vocabs)
                x_vector.append(torch.tensor(q_vec[:q_valid_lengths], dtype=torch.long))

                r_vec, r_valid_lengths = self.vectorizer.run(r, vocabs)
                y_vector.append(torch.tensor(r_vec[:r_valid_lengths], dtype=torch.long))
        features['x'] = x_vector
        features['y'] = y_vector
        logger.info("x shape: %s", len(x_vector))
        logger.info("y shape: %s", len(y_vector))
        return features

    def load(self, filename, vocabs):
        features = self.load_features(filename, vocabs)
        return ListDataset(features['x'], features['y'])


class ListDataset(Dataset):
    """Like the Provided TensorDataset but the tensors are stored in lists to facilitate different lengths."""
    def __init__(self, *tensors):
        if not all(len(tensors[0]) == len(tensor) for tensor in tensors):
            raise ValueError("All lists of paths to features must be equal length")
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])


def pad_batch(features):
    """Batch tensors on the fly.

    Input is a List[Tuple[torch.Tensor]] where the List is `batchsz` long and the Tuple is `[x, y]`

    Calculate the max length, create an empty batch of them and write the xs and ys into the batch

    return as a tuple of `[B, mxlen]` tensors
    """
    xs, ys = list(zip(*features))
    max_x = max(len(x) for x in xs)
    max_y = max(len(y) for y in ys)
    x_batch = torch.zeros((len(xs), max_x), dtype=torch.long)
    y_batch = torch.zeros((len(ys), max_y), dtype=torch.long)
    for i, x in enumerate(xs):
        x_batch[i, :len(x)] = x
    for i, y in enumerate(ys):
        y_batch[i, :len(y)] = y
    return (x_batch, y_batch)


# TODO: (blester) Consolidate all these disk tensors into some directory that is easy to remove
class TensorWordDiskDatasetReader(TensorWordDatasetReader):
    """Read words in and write the resulting tensor to disk."""

    def load_features(self, directory, vocabs):

        features = {}
        logger.info("Loading features from %s", os.path.basename(directory))
        x_paths = []
        y_paths = []
        for file in files_for_dir(directory):
            logger.info("Loading features for convo from %s", os.path.basename(file))
            file_prefix, _ = os.path.splitext(file)
            df = pd.read_csv(file, error_bad_lines=False)
            qs = df[self.query_field]
            ps = df[self.response_field]

            for i, (q, r) in enumerate(zip(qs, ps)):
                q = str(q).strip().split()
                r = str(r).strip().split()
                if q == '' or r == '':
                    continue
                q_vec, q_valid_lengths = self.vectorizer.run(q, vocabs)
                x = torch.tensor(q_vec, dtype=torch.long)
                x_path = f"{file_prefix}-{self.use_subword}" if self.use_subword is not None else file_prefix
                x_path = f"{x_path}-{i}-x.pt"
                logger.info("Saving x feature to %s", x_path)
                torch.save(x, x_path)
                x_paths.append(x_path)

                r_vec, r_valid_lengths = self.vectorizer.run(r, vocabs)
                y = torch.tensor(r_vec, dtype=torch.long)
                y_path = f"{file_prefix}-{self.use_subword}" if self.use_subword is not None else file_prefix
                y_path = f"{y_path}-{i}-y.pt"
                logger.info("Saving y feature to %s", y_path)
                torch.save(y, y_path)
                y_paths.append(y_path)

        features['x'] = x_paths
        features['y'] = y_paths
        return features

    def load(self, filename, vocab):
        features = self.load_features(filename, vocab)
        return DiskDataset(features['x'], features['y'])


class DiskDataset(Dataset):
    """Like the TensorDataset but it holds a list of paths. Calls to get item reads the tensor off disk."""
    def __init__(self, *paths):
        if not all(len(paths[0]) == len(path) for path in paths):
            raise ValueError("All lists of paths to features must be equal length")
        self.paths = paths

    def __getitem__(self, index):
        return tuple(torch.load(path[index]) for path in self.paths)

    def __len__(self):
        return len(self.paths[0])



def load_data(token_type, reader, dataset, file_key, vocabs, caching):
    cached_file = '{}-{}-paired.cache'.format(dataset[file_key], token_type)
    if caching and os.path.exists(cached_file):
        logger.info("Reloading %s from cached file [%s]", file_key, cached_file)
        loaded = torch.load(cached_file)
    else:
        loaded = reader.load(dataset[file_key], vocabs)
        logger.info("Caching %s to [%s]", file_key, cached_file)
        torch.save(loaded, cached_file)
    return loaded


READERS = {
    "packed": TensorWordDatasetReader,
    "jagged": TensorWordJaggedDatasetReader,
    "disk": TensorWordDiskDatasetReader
}


def create_reader(token_type, nctx, subword_model_file, subword_vocab_file, reader_type="packed"):
    if token_type == "chars":
        raise NotImplementedError("We do not currently support char tokens")
    elif token_type == "words":
        logger.info("Using word input")
        reader = READERS[reader_type](nctx)
    else:
        logger.info("Using subword ({}) input".format(token_type))
        Reader = READERS[reader_type]
        reader = Reader(nctx, token_type, subword_model_file, subword_vocab_file)
    return reader


def get_embed_and_vocab_cache(base_path, dataset_key, token_type):
    return os.path.join(base_path, 'preproc-{}-{}-paired.cache'.format(dataset_key, token_type))


def load_embed_and_vocab(token_type, reader, dataset, dataset_key, d_model, caching):
    base_path = os.path.dirname(dataset['train_file'])
    preproc_cache = get_embed_and_vocab_cache(base_path, dataset_key, token_type)
    if caching and os.path.exists(preproc_cache):
        logger.info("Loading cached preprocessing info [%s]", preproc_cache)
        preproc_data = torch.load(preproc_cache)

    else:
        vocab_sources = [dataset['train_file'], dataset['valid_file']]
        logger.info("Building Vocab from %s", vocab_sources)
        vocab = reader.build_vocab(vocab_sources)
        valid_num_words = reader.num_words[dataset['valid_file']]
        logger.info("Read vocabulary")
        embeddings = {}

        # If we are not using chars, then use 'x' for both input and output
        x_embedding = eight_mile.embeddings.load_embeddings('x',
                                                            dsz=d_model,
                                                            known_vocab=vocab,
                                                            embed_type='positional')
        logger.info(x_embedding)
        vocab = x_embedding['vocab']
        embedding = x_embedding['embeddings']

        preproc_data = {'vocabs': vocab, 'embeddings': embedding, 'valid_num_words': valid_num_words, "mxlen": reader.vectorizer.mxlen}
        logger.info("Saving preprocessing info [%s]", preproc_cache)
        torch.save(preproc_data, preproc_cache)
    return preproc_data


def checkpoint_for(model_base, epoch):
    return '{}-{}.pth'.format(model_base, epoch+1)


def rm_old_checkpoints(base_path, current_epoch, last_n=3):
    for i in range(0, current_epoch-last_n):
        checkpoint_i = checkpoint_for(base_path, i)
        if os.path.exists(checkpoint_i):
            logger.info("Removing: %s", checkpoint_i)
            os.remove(checkpoint_i)


def save_checkpoint(model: torch.nn.Module, model_base: str, count: int):

    checkpoint_name = checkpoint_for(model_base, count+1)
    # Its possible due to how its called that we might save the same checkpoint twice if we dont check first
    if os.path.exists(checkpoint_name):
        logger.info("Checkpoint already exists: %d", count+1)
        return
    logger.info("Creating checkpoint: %s", checkpoint_name)
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), checkpoint_name)
    else:
        torch.save(model.state_dict(), checkpoint_name)

    rm_old_checkpoints(model_base, count+1)


class Average(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class PairedLoss(nn.Module):
    """Provide a Triplet Loss using the reversed batch for negatives"""
    def __init__(self, model):
        super().__init__()
        self.score = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.model = model

    def forward(self, inputs, targets):
        # reverse the batch and use as a negative example
        neg = targets.flip(0)
        query = self.model.encode_query(inputs)
        response = self.model.encode_response(targets)
        neg_response = self.model.encode_response(neg)
        pos_score = self.score(query, response)
        neg_score = self.score(query, neg_response)
        score = neg_score - pos_score
        score = score.masked_fill(score < 0.0, 0.0).sum(0)
        return score


class AllLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.score = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.model = model

    def forward(self, inputs, targets):
        # These will get broadcast to [B, B, H]
        query = self.model.encode_query(inputs).unsqueeze(1)  # [B, 1, H]
        response = self.model.encode_response(targets).unsqueeze(0)  # [1, B, H]
        all_score = self.score(query, response)
        pos_score = torch.diag(all_score)
        loss = pos_score - vec_log_sum_exp(all_score, -1).squeeze()
        # Batch loss
        loss = torch.mean(loss)
        return -loss


class PairedModel(nn.Module):

    def __init__(self, embeddings, d_model, d_ff, dropout, num_heads, num_layers, stacking_layers=None, d_out=512, d_k=64):
        super().__init__()
        if stacking_layers is None:
            stacking_layers = [d_model] * 3

        stacking_layers = listify(stacking_layers)
        transformer = TransformerEncoderStack(num_heads=num_heads, d_model=d_model,
                                              pdrop=dropout, layers=num_layers, activation='gelu', d_ff=d_ff, d_k=d_k)
        self.attention_layer = MultiHeadedAttention(2, d_model, dropout, scale=True, d_k = d_k)
        self.transformer_layers = transformer
        self.embedding_layers = embeddings
        self.ff1 = DenseStack(d_model, stacking_layers + [d_out], activation='gelu')
        self.ff2 = DenseStack(d_model, stacking_layers + [d_out], activation='gelu')

    def encode_query(self, query):
        query_mask = (query != Offsets.PAD)
        query_length = query_mask.sum(-1)
        query_mask = query_mask.unsqueeze(1).unsqueeze(1)
        embedded = self.embedding_layers(query)
        encoded_query = self.transformer_layers((embedded, query_mask))
        encoded_query = self.attention_layer((encoded_query, encoded_query, encoded_query, query_mask))
        encoded_query = encoded_query.sum(1) / query_length.float().unsqueeze(1)
        encoded_query = self.ff1(encoded_query)
        return encoded_query

    def encode_response(self, response):
        response_mask = (response != Offsets.PAD)
        response_length = response_mask.sum(-1)
        response_mask = response_mask.unsqueeze(1).unsqueeze(1)
        embedded = self.embedding_layers(response)
        encoded_response = self.transformer_layers((embedded, response_mask))
        encoded_response = self.attention_layer((encoded_response, encoded_response, encoded_response, response_mask))
        encoded_response = encoded_response.sum(1) / response_length.float().unsqueeze(1)
        encoded_response = self.ff2(encoded_response)

        return encoded_response

    def forward(self, query, response):
        encoded_query = self.encode_query(query)
        encoded_response = self.encode_response(response)
        return encoded_query, encoded_response

    def create_loss(self, loss_type='all'):
        if loss_type == 'all':
            return AllLoss(self)
        return PairedLoss(self)

def create_model(embeddings, d_model, d_ff, dropout, num_heads, num_layers):

    model = PairedModel(embeddings, d_model, d_ff, dropout, num_heads, num_layers)
    logger.info(model)
    return model


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--dataset_key", type=str, default='auto', help="key from DATASETS global")
    parser.add_argument("--train_file", type=str, help='Optional file path to use for train file')
    parser.add_argument("--valid_file", type=str, help='Optional file path to use for valid file')
    parser.add_argument("--dataset_cache", type=str, default=os.path.expanduser('~/.bl-data'),
                        help="Path or url of the dataset cache")
    parser.add_argument("--cache_features", type=str2bool, default=True)
    parser.add_argument("--reader_type",
                        default="packed",
                        choices=("packed", "jagged", "disk"),
                        help=("How the tensor data is stored: "
                              "packed = two large dense tensors, "
                              "jagged = two lists of tensors that are padded as they are batched, "
                              "disk = two lists of paths that are read from disk on the fly"))
    parser.add_argument("--d_model", type=int, default=410, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2100, help="FFN dimension")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")
    parser.add_argument("--tokens", choices=["words", "bpe", "wordpiece"], default="wordpiece",
                        help="What tokens to use")
    parser.add_argument("--subword_model_file", type=str, help="If using subwords, pass this", default='bert-base-cased')
    parser.add_argument("--subword_vocab_file", type=str, help="If using subwords with separate vocab file, pass here")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Num warmup steps")
    parser.add_argument("--loss", type=str, default='all', choices=['triplet', 'all'])
    parser.add_argument("--update_steps", type=int, default=100, help="The number of steps to take before output a log message")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--distributed",
                        type=str2bool,
                        default=False,
                        help="Are we doing distributed training?")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Local rank for distributed training (-1 means use the environment variables to find)")

    args = parser.parse_args()

    if args.train_file and not args.valid_file:
        logger.error("If you provide a train_file, you must provide a valid_file")
        return

    if not args.train_file and args.valid_file:
        logger.error("If you provide a valid_file, you must also provide a train_file")
        return

    if args.basedir is None:
        args.basedir = 'paired-transformer-{}-{}-{}'.format(args.dataset_key, args.tokens, os.getpid())
    logging.basicConfig(
        format="%(name)s: %(levelname)s: %(message)s",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.info("Cache directory [%s]", args.dataset_cache)

    args.distributed = args.distributed or int(os.environ.get("WORLD_SIZE", 1)) > 1

    if args.distributed:
        if args.local_rank == -1:
            # https://github.com/kubeflow/pytorch-operator/issues/128
            # https://github.com/pytorch/examples/blob/master/imagenet/main.py
            logger.info("Setting local rank to RANK env variable")
            args.local_rank = int(os.environ['RANK'])
        logger.warning("Local rank (%d)", args.local_rank)
        # In an env like k8s with kubeflow each worker will only see a single gpu
        # with an id of 0. If the gpu count is 1 then we are probably in an env like
        # that so we should just use the first (and only) gpu avaiable
        if torch.cuda.device_count() == 1:
            torch.cuda.set_device(0)
            args.device = torch.device("cuda", 0)
        # This program assumes multiprocess/multi-device on a single node. Each
        # process gets a rank (via cli or ENV variable) and uses that rank to select
        # which gpu to use. This only makes sense on a single node, if you had 4
        # processes on 2 nodes where each node has 2 GPUs then the ranks would be
        # 0, 1, 2, 3 but the gpus numbers would be node 0: 0, 1 and node 1: 0, 1
        # and this assignment to gpu 3 would fail. On a single node with 4 processes
        # and 4 gpus the rank and gpu ids will align and this will work
        else:
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.train_file:
        dataset = {'train_file': args.train_file, 'valid_file': args.valid_file}
    else:
        try:
            dataset = DataDownloader(DATASETS[args.dataset_key], args.dataset_cache).download()
        except:
            dataset = DATASETS[args.dataset_key]
    reader = create_reader(args.tokens, args.nctx, args.subword_model_file, args.subword_vocab_file, args.reader_type)

    preproc_data = load_embed_and_vocab(args.tokens, reader, dataset, args.dataset_key, args.d_model, args.cache_features)

    vocabs = preproc_data['vocabs']
    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs, os.path.join(args.basedir, 'vocabs.json'))
    embeddings = preproc_data['embeddings']
    valid_num_words = preproc_data['valid_num_words']
    logger.info("Loaded embeddings")

    train_set = load_data(args.tokens, reader, dataset, 'train_file', vocabs, args.cache_features)
    valid_set = load_data(args.tokens, reader, dataset, 'valid_file', vocabs, args.cache_features)
    # logger.info("valid. tokens [%s], valid. words [%s]", valid_set.tensors[-1].numel(), valid_num_words)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None
    collate_fn = pad_batch if isinstance(train_set, ListDataset) else None
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=args.batch_size, shuffle=(not args.distributed), collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info("Loaded datasets")

    model = create_model(embeddings, d_model=args.d_model, d_ff=args.d_ff, dropout=args.dropout,
                         num_heads=args.num_heads, num_layers=args.num_layers)
    model.to(args.device)
    loss_function = model.create_loss(args.loss)
    loss_function.to(args.device)

    logger.info("Loaded model and loss")

    steps_per_epoch = len(train_loader)
    update_on = steps_per_epoch // args.update_steps
    cosine_decay = CosineDecaySchedulerPyTorch(len(train_loader) * args.epochs, lr=args.lr)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, cosine_decay, lr=args.lr)

    global_step = 0
    start_epoch = 0
    if args.restart_from:
        model.load_state_dict(torch.load(args.restart_from))
        start_epoch = int(args.restart_from.split("-")[-1].split(".")[0]) - 1
        global_step = (start_epoch+1) * steps_per_epoch
        logger.info("Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d, epoch=%d",
                    args.restart_from, global_step, start_epoch+1)
    optimizer = OptimizerManager(model, global_step, optim='adam', lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay)
    logger.info("Model has {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Prepare model for distributed training if needed
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        logger.info("Model located on %d", args.local_rank)

    model_base = os.path.join(args.basedir, 'checkpoint')

    # This is the training loop
    steps = 0
    for epoch in range(start_epoch, args.epochs):
        avg_loss = Average('average_train_loss')
        metrics = {}
        optimizer.zero_grad()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        start = time.time()
        model.train()
        for i, batch in enumerate(train_loader):
            steps += 1
            x, y = batch
            inputs = x.to(args.device)
            labels = y.to(args.device)
            loss = loss_function(inputs, labels)
            loss.backward()
            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % update_on == 0:
                logging.info(avg_loss)
                if args.local_rank < 1:
                    save_checkpoint(model, model_base, steps)

        # How much time elapsed in minutes
        elapsed = (time.time() - start)/60
        train_avg_loss = avg_loss.avg
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_loss'] = train_avg_loss
        avg_valid_loss = Average('average_valid_loss')
        start = time.time()
        model.eval()
        for batch in valid_loader:
            with torch.no_grad():
                x, y = batch
                inputs = x.to(args.device)
                labels = y.to(args.device)
                loss = loss_function(inputs, labels)
                avg_valid_loss.update(loss.item())

        valid_avg_loss = avg_valid_loss.avg

        elapsed = (time.time() - start)/60
        metrics['valid_elapsed_min'] = elapsed

        metrics['average_valid_loss'] = valid_avg_loss
        logger.info(metrics)
        if args.local_rank < 1:
            save_checkpoint(model, model_base, steps)



if __name__ == "__main__":
    train()

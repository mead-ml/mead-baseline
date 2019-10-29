import logging
import pickle
import time
import os
from argparse import ArgumentParser
import baseline
from eight_mile.utils import str2bool, write_json
import eight_mile.embeddings
from eight_mile.tf.embeddings import *
from eight_mile.optz import CompositeLRScheduler, WarmupLinearScheduler, CosineDecayScheduler
from eight_mile.tf.optz import *
from baseline.tf.lm import TransformerLanguageModel
from eight_mile.tf.layers import set_tf_log_level
from baseline.vectorizers import Token1DVectorizer, AbstractVectorizer
from mead.downloader import DataDownloader

import codecs
from collections import Counter

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

NUM_PREFETCH = 2
SHUF_BUF_SZ = 5000
"""Pre-train a Transformer model in TensorFlow eager

This file uses Baseline to train a Transformer with TensorFlow on multiple GPUs.
It is inspired by: https://github.com/huggingface/naacl_transfer_learning_tutorial/blob/master/pretraining_train.py

"""
DATASETS = {
    "ptb": {
        "train_file": "train.txt",
        "valid_file": "valid.txt",
        "test_file": "test.txt",
        "download": "https://www.dropbox.com/s/5g8en2jc9951omu/ptb.tar.gz?dl=1",
        "sha1": "56aacd9bd3aeffb34a9536e8de2341a8d6770f7b"
    },
    "wikitext-2": {
        "train_file": "train.txt",
        "valid_file": "valid.txt",
        "test_file": "test.txt",
        "download": "https://www.dropbox.com/s/q4i2vxw1nkhsk8g/wikitext-2.tar.gz?dl=1"
    },
    "wikitext-2-raw": {
        "train_file": "wikitext-2-raw/wiki.train.raw",
        "valid_file": "wikitext-2-raw/wiki.valid.raw",
        "test_file": "wikitext-2-raw/wiki.test.raw",
        "download": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
    },
    "wikitext-103": {
        "train_file": "wikitext-103/wiki.train.tokens",
        "valid_file": "wikitext-103/wiki.valid.tokens",
        "test_file": "wikitext-103/wiki.test.tokens",
        "download": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
    },
    "wikitext-103-raw": {
        "train_file": "wikitext-103/wiki.train.raw",
        "valid_file": "wikitext-103/wiki.valid.raw",
        "test_file": "wikitext-103/wiki.test.raw",
        "download": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
    }
}

class BPEVectorizer1D(AbstractVectorizer):
    """Define a Baseline Vectorizer for BPE using fastBPE (https://github.com/glample/fastBPE)

    If you use tokens=bpe, this vectorizer is used, and so then there is a
    dependency on fastBPE

    To use BPE, we assume that a Dictionary of codes and vocab was already created

    """
    def __init__(self, **kwargs):
        """Loads a BPE tokenizer"""
        super(BPEVectorizer1D, self).__init__(kwargs.get('transform_fn'))
        from fastBPE import fastBPE
        self.max_seen = 128
        self.model_file = kwargs.get('model_file')
        self.vocab_file = kwargs.get('vocab_file')
        self.tokenizer = fastBPE(self.model_file, self.vocab_file)
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
            if t == '<unk>':
                yield Offsets.UNK
            if t == '<eos>':
                yield Offsets.EOS
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


class TensorDatasetReaderBase(object):
    """Provide a base-class to do operations that are independent of token representation
    """
    def __init__(self, nctx, vectorizers):
        self.vectorizers = vectorizers
        self.nctx = nctx
        self.num_words = {}

    def build_vocab(self, files):
        vocabs = {k: Counter() for k in self.vectorizers.keys()}

        for file in files:
            if file is None:
                continue
            self.num_words[file] = 0
            with codecs.open(file, encoding='utf-8', mode='r') as f:
                sentences = []
                for line in f:
                    split_sentence = line.split() + ['<EOS>']
                    self.num_words[file] += len(split_sentence)
                    sentences += split_sentence
                for k, vectorizer in self.vectorizers.items():
                    vocabs[k].update(vectorizer.count(sentences))
        return vocabs

    def load_features(self, filename, vocabs):

        features = dict()
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            sentences = []
            for line in f:
                sentences += line.strip().split() + ['<EOS>']
            for k, vectorizer in self.vectorizers.items():
                vec, valid_lengths = vectorizer.run(sentences, vocabs[k])
                features[k] = vec[:valid_lengths]
                shp = list(vectorizer.get_dims())
                shp[0] = valid_lengths
                features['{}_dims'.format(k)] = tuple(shp)
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
        else:
            vectorizer = Token1DVectorizer(transform_fn=baseline.lowercase)
        super().__init__(nctx, {'x': vectorizer})

    def build_vocab(self, files):
        """Read the vocab file to get the tokens

        :param files:
        :return:
        """
        if self.use_subword is not None:
            super(TensorWordDatasetReader, self).build_vocab(files)
            return {'x': self.vectorizers['x'].vocab}
        return super(TensorWordDatasetReader, self).build_vocab(files)

    def load(self, filename, vocabs, batch_size):
        features = self.load_features(filename, vocabs)
        x_tensor = features['x']
        num_sequences_word = (np.prod(x_tensor.shape) // self.nctx) * self.nctx
        x_tensor = x_tensor[:num_sequences_word].reshape(-1, self.nctx)
        dataset = tf.data.Dataset.from_tensor_slices((x_tensor, x_tensor))
        dataset = dataset.shuffle(buffer_size=SHUF_BUF_SZ)
        dataset = dataset.batch(batch_size, True)
        dataset = dataset.map(lambda x, y: (x, y))
        dataset = dataset.prefetch(NUM_PREFETCH)
        return dataset

def save_cache(cached_file, thing, protocol='pickle'):
    with open(cached_file, 'wb') as f:
        pickle.dump(thing, f)

def load_cache(cached_file):
    with open(cached_file, 'rb') as f:
        loaded = pickle.load(cached_file)
        return loaded

def load_data(token_type, reader, dataset, file_key, vocabs, batch_size, caching):
    cached_file = '{}-{}.tf-cache'.format(dataset[file_key], token_type)
    if caching and os.path.exists(cached_file):
        logger.info("Reloading %s from cached file [%s]", file_key, cached_file)
        loaded = load_cache(cached_file)

    else:
        loaded = reader.load(dataset[file_key], vocabs, batch_size)
        logger.info("Caching %s to [%s]", file_key, cached_file)
        #save_cache(cached_file, loaded)
    return loaded


def create_reader(token_type, nctx, subword_model_file, subword_vocab_file):
    if token_type == "words":
        logger.info("Using word input")
        reader = TensorWordDatasetReader(nctx)
    else:
        logger.info("Using subword ({}) input".format(token_type))
        reader = TensorWordDatasetReader(nctx, token_type, subword_model_file, subword_vocab_file)
    return reader


def get_embed_and_vocab_cache(base_path, dataset_key, token_type):
    return os.path.join(base_path, 'preproc-{}-{}.tf-cache'.format(dataset_key, token_type))


def load_embed_and_vocab(token_type, reader, dataset, dataset_key, d_model, caching):
    base_path = os.path.dirname(dataset['train_file'])
    preproc_cache = get_embed_and_vocab_cache(base_path, dataset_key, token_type)
    if caching and os.path.exists(preproc_cache):
        logger.info("Loading cached preprocessing info [%s]", preproc_cache)
        preproc_data = load_cache(preproc_cache)

    else:
        vocab_sources = [dataset['train_file'], dataset['valid_file']]
        vocabs = reader.build_vocab(vocab_sources)
        valid_num_words = reader.num_words[dataset['valid_file']]
        train_num_words = reader.num_words[dataset['train_file']]
        logger.info("Read vocabulary")
        embeddings = {}

        # If we are not using chars, then use 'x' for both input and output
        tgt_key = 'x'
        x_embedding = eight_mile.embeddings.load_embeddings('x',
                                                            dsz=d_model,
                                                            known_vocab=vocabs['x'],
                                                            embed_type='positional')
        vocabs['x'] = x_embedding['vocab']
        embeddings['x'] = x_embedding['embeddings']

        preproc_data = {'vocabs': vocabs, 'embeddings': embeddings, 'train_num_words': train_num_words, 'valid_num_words': valid_num_words, 'tgt_key': tgt_key}
        logger.info("Saving preprocessing info [%s]", preproc_cache)
        ##save_cache(preproc_cache, preproc_data)

    return preproc_data


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


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--dataset_key", type=str, default='wikitext-2', help="key from DATASETS global")
    parser.add_argument("--train_file", type=str, help='Optional file path to use for train file')
    parser.add_argument("--valid_file", type=str, help='Optional file path to use for valid file')
    parser.add_argument("--dataset_cache", type=str, default=os.path.expanduser('~/.bl-data'),
                        help="Path or url of the dataset cache")
    parser.add_argument("--cache_features", type=str2bool, default=True)
    parser.add_argument("--d_model", type=int, default=410, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2100, help="FFN dimension")
    parser.add_argument("--num_heads", type=int, default=10, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=16, help="Number of layers")
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")
    parser.add_argument("--tokens", choices=["words", "chars", "bpe", "wordpiece"], default="wordpiece",
                        help="What tokens to use")
    parser.add_argument("--subword_model_file", type=str, help="If using subwords, pass this", default='bert-base-cased')
    parser.add_argument("--subword_vocab_file", type=str, help="If using subwords with separate vocab file, pass here")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Num training epochs")
    #parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Num warmup steps")
    parser.add_argument("--mlm", type=str2bool, default=False, help="Use Masked Language Model (MLM) objective")
    parser.add_argument("--distributed",
                        type=str2bool,
                        default=False,
                        help="Are we doing distributed training?")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Local rank for distributed training (-1 means use the environment variables to find)")
    parser.add_argument("--chars_per_word",
                        type=int,
                        default=40,
                        help="How many max characters per word")
    parser.add_argument('--tf_ll', help='TensorFlow Log level', type=str, default='warn')
    args = parser.parse_args()
    set_tf_log_level(args.tf_ll)

    if args.train_file and not args.valid_file:
        logger.error("If you provide a train_file, you must provide a valid_file")
        return

    if not args.train_file and args.valid_file:
        logger.error("If you provide a valid_file, you must also provide a train_file")
        return

    if args.basedir is None:
        args.basedir = 'transformer-{}-{}-{}'.format(args.dataset_key, args.tokens, os.getpid())
    #logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("Cache directory [%s]", args.dataset_cache)

    # TODO:
    #args.distributed = args.distributed or int(os.environ.get("WORLD_SIZE", 1)) > 1

    #if args.distributed:
    #    if args.local_rank == -1:
    #        # https://github.com/kubeflow/pytorch-operator/issues/128
    #        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    #        logger.info("Setting local rank to RANK env variable")
    #        args.local_rank = int(os.environ['RANK'])
    #    logger.warning("Local rank (%d)", args.local_rank)
    #    torch.cuda.set_device(args.local_rank)
    #    args.device = torch.device("cuda", args.local_rank)
    #    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.train_file:
        dataset = {'train_file': args.train_file, 'valid_file': args.valid_file}
    else:
        dataset = DataDownloader(DATASETS[args.dataset_key], args.dataset_cache).download()
    reader = create_reader(args.tokens, args.nctx, args.subword_model_file, args.subword_vocab_file)

    preproc_data = load_embed_and_vocab(args.tokens, reader, dataset, args.dataset_key, args.d_model, args.cache_features)

    vocabs = preproc_data['vocabs']
    vocab_size = len(vocabs['x'])
    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs['x'], os.path.join(args.basedir, 'vocabs.json'))
    embeddings = preproc_data['embeddings']
    ## Approx
    steps_per_epoch = preproc_data['train_num_words'] // args.nctx // args.batch_size
    valid_num_words = preproc_data['valid_num_words']
    tgt_key = preproc_data['tgt_key']
    logger.info("Loaded embeddings")

    train_set = load_data(args.tokens, reader, dataset, 'train_file', vocabs, args.batch_size, args.cache_features)
    valid_set = load_data(args.tokens, reader, dataset, 'valid_file', vocabs, args.batch_size, args.cache_features)
    logger.info("valid. words [%s]", valid_num_words)
    logger.info("Loaded datasets")

    model = TransformerLanguageModel.create(embeddings,
                                            hsz=args.d_model,
                                            d_ff=args.d_ff,
                                            tie_weights=False,
                                            dropout=args.dropout,
                                            gpu=False,
                                            num_heads=args.num_heads,
                                            layers=args.num_layers,
                                            src_keys=['x'], tgt_key=tgt_key)
    def loss_function(logits, y):
        vsz = vocab_size
        targets = tf.reshape(y, [-1])
        bt_x_v = tf.nn.log_softmax(tf.reshape(logits, [-1, vsz]), axis=-1)
        one_hots = tf.one_hot(targets, vsz)
        example_loss = -tf.reduce_sum(one_hots * bt_x_v, axis=-1)
        loss = tf.reduce_mean(example_loss)
        return loss

    def lm_loss(model, x, y):
        labels = tf.transpose(y, perm=[1, 0])
        output = model(x)[0]
        logits = tf.transpose(output, perm=[1, 0, 2])
        shift_logits = logits[:-1]
        shift_labels = labels[1:]
        return loss_function(shift_logits, shift_labels)


    logger.info("Loaded model and loss")

    update_on = steps_per_epoch // 10
    start_epoch = 0
    optimizer = EagerOptimizer(lm_loss, tf.optimizers.Adam(lr=args.lr))
    _checkpoint = tf.train.Checkpoint(optimizer=optimizer.optimizer, model=model)
    model_base = os.path.join(args.basedir, 'checkpoint')
    checkpoint_dir = '{}-{}'.format(model_base, os.getpid())
    checkpoint_manager = tf.train.CheckpointManager(_checkpoint,
                                                    directory=checkpoint_dir,
                                                    max_to_keep=5)
    # This is the training loop
    for epoch in range(start_epoch, args.epochs):
        avg_loss = Average('average_train_loss')
        metrics = {}

        start = time.time()
        SET_TRAIN_FLAG(True)
        for i, batch in enumerate(train_set):
            x, y = batch
            loss_value = optimizer.update(model, {'x': x}, y)
            avg_loss.update(loss_value.numpy())
            if (i + 1) % update_on == 0:
                print(avg_loss)

        # How much time elapsed in minutes
        elapsed = (time.time() - start)/60
        train_token_loss = avg_loss.avg
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        train_token_ppl = math.exp(train_token_loss)
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_loss'] = train_token_loss
        metrics['train_ppl'] = train_token_ppl
        avg_valid_loss = Average('average_valid_loss')
        start = time.time()
        SET_TRAIN_FLAG(False)
        for batch in valid_set:
            x, y = batch
            loss_value = lm_loss(model, {'x': x}, y)
            avg_valid_loss.update(loss_value.numpy())

        valid_token_loss = avg_valid_loss.avg
        valid_token_ppl = math.exp(valid_token_loss)

        elapsed = (time.time() - start)/60
        metrics['valid_elapsed_min'] = elapsed

        metrics['average_valid_loss'] = valid_token_loss
        if args.tokens in ['bpe', 'wordpiece']:
            metrics['valid_token_ppl'] = valid_token_ppl
            metrics['average_valid_word_ppl'] = math.exp(valid_token_loss * valid_set.tensors[-1].numel() / valid_num_words)
        else:
            metrics['average_valid_word_ppl'] = valid_token_ppl
        print(metrics)

        if args.local_rank < 1:
            # Should probably do this more often
            checkpoint_manager.save()


if __name__ == "__main__":
    train()


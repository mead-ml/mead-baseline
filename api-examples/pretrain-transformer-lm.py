import logging
import time
import os
from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from baseline.pytorch.lm import TransformerLanguageModel
from baseline.utils import str2bool, write_json
import baseline.embeddings
from baseline.pytorch.embeddings import *
from baseline.pytorch.optz import *
from baseline.vectorizers import Char2DVectorizer, Token1DVectorizer, AbstractVectorizer
from mead.downloader import DataDownloader

import codecs
from collections import Counter

logger = logging.getLogger(__file__)

"""Pre-train a Transformer model in PyTorch

This file uses Baseline to train a Transformer with PyTorch on multiple GPUs.
It is inspired by: https://github.com/huggingface/naacl_transfer_learning_tutorial/blob/master/pretraining_train.py
This pretraining module has multiple configurations that allow it to support

  * 3 types of pre-training tokenization
    - word
    - subword (based on BERT tokenizer)
    - ELMo (Kim et al 2015) char method
  * pretraining on several datasets including PTB, Wikitext 2 (including raw) and Wikitext 103 (including raw).

If you use subwords for tokens, this code requires bert_pretrained_pytorch.  Otherwise, it depends only on six, numpy,
pytorch, and baseline.

Because we are trying to pretrain a language model so we can do better on downstream tasks, it probably makes more
sense to train on a full word model, not a model where rare words have already been replaced.

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

X_CHAR_EMBEDDINGS = {
    "dsz": 16,
    "wsz": 128,
    "embed_type": "positional-char-conv",
    "keep_unused": True,
    "cfiltsz": [
        [1, 32],
        [2, 32],
        [3, 64],
        [4, 128],
        [5, 256],
        [6, 512],
        [7, 1024]
    ],
    "gating": "highway",
    "num_gates": 2,
    "projsz": 512
}

BERT_TOKENIZER = None


class WordPieceVectorizer1D(AbstractVectorizer):
    """Define a Baseline Vectorizer that can do WordPiece with BERT tokenizer

    If you use tokens=subword, this vectorizer is used, and so then there is
    a dependency on bert_pretrained_pytorch
    """

    def __init__(self, **kwargs):
        """Loads a BertTokenizer using bert_pretrained_pytorch

        :param kwargs:
        """
        super(WordPieceVectorizer1D, self).__init__(kwargs.get('transform_fn'))
        from pytorch_pretrained_bert import BertTokenizer
        self.max_seen = 128
        handle = kwargs.get('embed_file')
        self.tokenizer = BertTokenizer.from_pretrained(handle, do_lower_case=False)
        self.mxlen = kwargs.get('mxlen', -1)

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
    def __init__(self, nctx, use_wordpiece=False):
        """Create a reader with a context window that reads words

        :param nctx: The context window length
        """
        self.use_wordpiece = use_wordpiece
        if use_wordpiece:
            vectorizer = WordPieceVectorizer1D(embed_file='bert-base-cased')
        else:
            vectorizer = Token1DVectorizer(transform_fn=baseline.lowercase)
        super(TensorWordDatasetReader, self).__init__(nctx, {'x': vectorizer})

    def build_vocab(self, files):
        """Read the vocab file to get the tokens

        :param files:
        :return:
        """
        if self.use_wordpiece:
            super(TensorWordDatasetReader, self).build_vocab(files)
            return {'x': self.vectorizers['x'].tokenizer.vocab}
        return super(TensorWordDatasetReader, self).build_vocab(files)

    def load(self, filename, vocabs):
        features = self.load_features(filename, vocabs)
        x_tensor = torch.tensor(features['x'], dtype=torch.long)
        num_sequences_word = (x_tensor.size(0) // self.nctx) * self.nctx
        x_tensor = x_tensor.narrow(0, 0, num_sequences_word).view(-1, self.nctx)
        return TensorDataset(x_tensor, x_tensor)


class TensorCharDatasetReader(TensorDatasetReaderBase):
    """TensorCharDatasetReader reads in a vocab and then a dataset and returns as a `dict` of `string` to `ndarray`
    """
    def __init__(self, nctx, chars_per_word):
        y_vectorizer = Token1DVectorizer(transform_fn=baseline.lowercase)
        x_vectorizer = Char2DVectorizer(mxwlen=chars_per_word)
        super(TensorCharDatasetReader, self).__init__(nctx, {'x': x_vectorizer, 'y': y_vectorizer})
        self.chars_per_word = chars_per_word

    def load(self, filename, vocabs):
        features = self.load_features(filename, vocabs)
        y_tensor = torch.tensor(features['y'], dtype=torch.long)
        num_sequences_word = (y_tensor.size(0) // self.nctx) * self.nctx
        y_tensor = y_tensor.narrow(0, 0, num_sequences_word).view(-1, self.nctx)

        x_dataset = torch.tensor(features['x'], dtype=torch.long)
        x_tensor = torch.tensor(x_dataset, dtype=torch.long)
        x_tensor = x_tensor.narrow(0, 0, num_sequences_word)
        x_tensor = x_tensor.view(-1, self.nctx, self.chars_per_word)
        return TensorDataset(x_tensor, y_tensor)


def load_data(token_type, reader, dataset, file_key, vocabs, caching):
    cached_file = '{}-{}.cache'.format(dataset[file_key], token_type)
    if caching and os.path.exists(cached_file):
        logger.info("Reloading %s from cached file [%s]", file_key, cached_file)
        loaded = torch.load(cached_file)
    else:
        loaded = reader.load(dataset[file_key], vocabs)
        logger.info("Caching %s to [%s]", file_key, cached_file)
        torch.save(loaded, cached_file)
    return loaded


def create_reader(token_type, nctx, chars_per_word):
    if token_type == "chars":
        logger.info("Using character input")
        reader = TensorCharDatasetReader(nctx, chars_per_word)
    elif token_type == "words":
        logger.info("Using word input")
        reader = TensorWordDatasetReader(nctx, False)
    else:
        logger.info("Using subword (wordpiece) input")
        reader = TensorWordDatasetReader(nctx, True)
    return reader


def get_embed_and_vocab_cache(base_path, dataset_key, token_type):
    return os.path.join(base_path, 'preproc-{}-{}.cache'.format(dataset_key, token_type))


def load_embed_and_vocab(token_type, reader, dataset, dataset_key, d_model, caching):
    base_path = os.path.dirname(dataset['train_file'])
    preproc_cache = get_embed_and_vocab_cache(base_path, dataset_key, token_type)
    if caching and os.path.exists(preproc_cache):
        logger.info("Loading cached preprocessing info [%s]", preproc_cache)
        preproc_data = torch.load(preproc_cache)

    else:
        vocab_sources = [dataset['train_file'], dataset['valid_file']]
        vocabs = reader.build_vocab(vocab_sources)
        valid_num_words = reader.num_words[dataset['valid_file']]
        logger.info("Read vocabulary")
        embeddings = {}

        # If we are not using chars, then use 'x' for both input and output
        tgt_key = 'x'
        if token_type == 'chars':
            # Write JSON file here and skip this step the second time
            x_embedding = baseline.embeddings.load_embeddings('x', known_vocab=vocabs['x'], **X_CHAR_EMBEDDINGS)
            vocabs['x'] = x_embedding['vocab']

            y_embedding = baseline.embeddings.load_embeddings('y', dsz=1, known_vocab=vocabs['y'])
            vocabs['y'] = y_embedding['vocab']

            embeddings['x'] = x_embedding['embeddings']
            embeddings['y'] = y_embedding['embeddings']
            tgt_key = 'y'
        else:
            x_embedding = baseline.embeddings.load_embeddings('x',
                                                              dsz=d_model,
                                                              known_vocab=vocabs['x'],
                                                              embed_type='positional')
            vocabs['x'] = x_embedding['vocab']
            embeddings['x'] = x_embedding['embeddings']

        preproc_data = {'vocabs': vocabs, 'embeddings': embeddings, 'valid_num_words': valid_num_words, 'tgt_key': tgt_key}
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
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")
    parser.add_argument("--tokens", choices=["words", "chars", "subwords"], default="subwords",
                        help="What tokens to use")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Num warmup steps")
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
    parser.add_argument("--chars_per_word",
                        type=int,
                        default=40,
                        help="How many max characters per word")

    args = parser.parse_args()

    if args.train_file and not args.valid_file:
        logger.error("If you provide a train_file, you must provide a valid_file")
        return

    if not args.train_file and args.valid_file:
        logger.error("If you provide a valid_file, you must also provide a train_file")
        return

    if args.basedir is None:
        args.basedir = 'transformer-{}-{}-{}'.format(args.dataset_key, args.tokens, os.getpid())
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("Cache directory [%s]", args.dataset_cache)

    args.distributed = args.distributed or int(os.environ.get("WORLD_SIZE", 1)) > 1

    if args.distributed:
        if args.local_rank == -1:
            # https://github.com/kubeflow/pytorch-operator/issues/128
            # https://github.com/pytorch/examples/blob/master/imagenet/main.py
            logger.info("Setting local rank to RANK env variable")
            args.local_rank = int(os.environ['RANK'])
        logger.warning("Local rank (%d)", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.train_file:
        dataset = {'train_file': args.train_file, 'valid_file': args.valid_file}
    else:
        dataset = DataDownloader(DATASETS[args.dataset_key], args.dataset_cache).download()
    reader = create_reader(args.tokens, args.nctx, args.chars_per_word)

    preproc_data = load_embed_and_vocab(args.tokens, reader, dataset, args.dataset_key, args.d_model, args.cache_features)

    vocabs = preproc_data['vocabs']
    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs['x'], os.path.join(args.basedir, 'vocabs.json'))
    embeddings = preproc_data['embeddings']
    valid_num_words = preproc_data['valid_num_words']
    tgt_key = preproc_data['tgt_key']
    logger.info("Loaded embeddings")

    train_set = load_data(args.tokens, reader, dataset, 'train_file', vocabs, args.cache_features)
    valid_set = load_data(args.tokens, reader, dataset, 'valid_file', vocabs, args.cache_features)
    logger.info("valid. tokens [%s], valid. words [%s]", valid_set.tensors[-1].numel(), valid_num_words)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=args.batch_size, shuffle=(not args.distributed))

    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set) if args.distributed else None
    valid_loader = DataLoader(valid_set, sampler=valid_sampler, batch_size=args.batch_size, shuffle=False)

    logger.info("Loaded datasets")

    model = TransformerLanguageModel.create(embeddings,
                                            hsz=args.d_model,
                                            d_ff=args.d_ff,
                                            tie_weights=(args.tokens != 'chars'),
                                            dropout=args.dropout,
                                            gpu=False,
                                            num_heads=args.num_heads,
                                            layers=args.num_layers,
                                            src_keys=['x'], tgt_key=tgt_key)
    model.to(args.device)
    loss_function = model.create_loss()
    loss_function.to(args.device)

    logger.info("Loaded model and loss")

    steps_per_epoch = len(train_loader)
    valid_steps_per_epoch = len(valid_loader)
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
    logger.info("Model has %s parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Prepare model for distributed training if needed
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        logger.info("Model located on %d", args.local_rank)

    # This is the training loop
    for epoch in range(start_epoch, args.epochs):
        avg_loss = Average('average_train_loss')
        metrics = {}
        optimizer.zero_grad()

        if args.distributed:
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

        start = time.time()
        model.train()
        for i, batch in enumerate(train_loader):
            x, y = batch
            inputs = {'x': x.to(args.device)}
            labels = y.to(args.device).transpose(0, 1).contiguous()
            logits = model(inputs, None)[0].transpose(0, 1).contiguous()
            shift_logits = logits[:-1]
            shift_labels = labels[1:]
            loss = loss_function(shift_logits, shift_labels)
            loss.backward()
            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % 100 == 0:
                logging.info(avg_loss)

        # How much time elapsed in minutes
        elapsed = (time.time() - start)/60
        train_token_loss = avg_loss.avg
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        train_token_ppl = math.exp(train_token_loss)
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_loss'] = train_token_loss
        metrics['train_ppl'] = train_token_ppl
        model_base = os.path.join(args.basedir, 'checkpoint')
        avg_valid_loss = Average('average_valid_loss')
        model.eval()
        for batch in valid_loader:
            with torch.no_grad():
                x, y = batch
                inputs = {'x': x.to(args.device)}
                labels = y.to(args.device).transpose(0, 1).contiguous()
                logits = model(inputs, None)[0].transpose(0, 1).contiguous()
                shift_logits = logits[:-1]
                shift_labels = labels[1:]
                loss = loss_function(shift_logits, shift_labels)
                avg_valid_loss.update(loss.item())

        valid_token_loss = avg_valid_loss.avg
        valid_token_ppl = math.exp(valid_token_loss)

        elapsed = (time.time() - start)/60
        metrics['valid_elapsed_min'] = elapsed

        metrics['average_valid_loss'] = valid_token_loss
        if args.tokens == 'subwords':
            metrics['valid_token_ppl'] = valid_token_ppl
            metrics['average_valid_word_ppl'] = math.exp(valid_token_loss * valid_set.tensors[-1].numel() / valid_num_words)
        else:
            metrics['average_valid_word_ppl'] = valid_token_ppl
        logger.info(metrics)

        if args.local_rank < 1:

            # Should probably do this more often
            checkpoint_name = checkpoint_for(model_base, epoch+1)
            logger.info("Creating checkpoint: %s", checkpoint_name)
            torch.save(model.state_dict(), checkpoint_name)
            rm_old_checkpoints(model_base, epoch+1)


if __name__ == "__main__":
    train()


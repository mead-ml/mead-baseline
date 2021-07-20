from baseline.pytorch.torchy import vec_log_sum_exp
from baseline.pytorch.seq2seq import Seq2SeqModel
from eight_mile.utils import str2bool, write_yaml, read_yaml, Offsets
from eight_mile.pytorch.layers import *
from eight_mile.optz import create_lr_scheduler
import baseline.pytorch.embeddings
import baseline.embeddings
import random
from eight_mile.progress import create_progress_bar
from torch.utils.data.dataset import IterableDataset, TensorDataset
from baseline.vectorizers import Token1DVectorizer, BPEVectorizer1D, Char2DVectorizer, WordpieceVectorizer1D
import codecs
from collections import Counter
import glob
import json



class TensorDatasetReaderBase:
    """Provide a base-class to do operations that are independent of token representation
    """
    def __init__(self, nctx, vectorizers):
        self.vectorizers = vectorizers
        self.nctx = nctx
        self.num_words = {}

    def build_vocab(self, files):
        vocabs = {k: Counter({'[CLS]': 1}) for k in self.vectorizers.keys()}

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
    def __init__(self, nctx, use_subword=None, model_file=None, vocab_file=None, special_tokens=None):
        """Create a reader with a context window that reads words

        :param nctx: The context window length
        :param use_subword: If this is not none, it should be either 'bpe' or 'wordpiece'
        """
        self.use_subword = use_subword

        if self.use_subword == 'bpe':
            vectorizer = BPEVectorizer1D(model_file=model_file, vocab_file=vocab_file)
        elif self.use_subword == 'wordpiece':
            vectorizer = WordpieceVectorizer1D(embed_file=model_file, vocab_file=vocab_file,
                                               special_tokens=special_tokens)
        else:
            vectorizer = Token1DVectorizer(transform_fn=baseline.lowercase)
        super().__init__(nctx, {'x': vectorizer})

    def build_vocab(self, files):
        """Read the vocab file to get the tokens

        :param files:
        :return:
        """
        if self.use_subword is not None:
            super().build_vocab(files)
            return {'x': self.vectorizers['x'].vocab}
        return super().build_vocab(files)

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
        super().__init__(nctx, {'x': x_vectorizer, 'y': y_vectorizer})
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


def load_data_caching(token_type, reader, dataset, file_key, vocabs, caching, logger):
    cached_file = '{}-{}.cache'.format(dataset[file_key], token_type)
    if caching and os.path.exists(cached_file):
        logger.info("Reloading %s from cached file [%s]", file_key, cached_file)
        loaded = torch.load(cached_file)
    else:
        # if build_vocab() was never run, need to run it to set reader.vectorizer.mxlen correctly
        if reader.vectorizers['x'].mxlen == -1:
            _ = reader.build_vocab([dataset[file_key]])
        loaded = reader.load(dataset[file_key], vocabs)
        logger.info("Caching %s to [%s]", file_key, cached_file)
        torch.save(loaded, cached_file)
    return loaded


class MultiFileLoader(IterableDataset):

    def __init__(
            self, directory, pattern, vocabs, src_vectorizer, tgt_vectorizer, last_turn_only=False,
            distribute=True, shuffle=True, record_keys=[]
    ):
        super().__init__()
        self.record_keys = record_keys
        self.src_vectorizer = src_vectorizer
        self.tgt_vectorizer = tgt_vectorizer
        self.pattern = pattern
        self.directory = directory
        self.vocab = vocabs
        self.samples = 0
        self.rank = 0
        self.world_size = 1
        self.shuffle = shuffle
        self.last_turn_only = last_turn_only
        self.distribute = distribute
        if torch.distributed.is_initialized() and distribute:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

        if os.path.exists(f"{directory}/md.yml"):
            f = read_yaml(f"{directory}/md.yml")
            self.samples = f['num_samples']
        else:
            files = list(glob.glob(f"{directory}/{self.pattern}"))
            pg = create_progress_bar(len(files))
            for file in pg(files):
                with open(file) as rf:
                    for _ in rf:
                        self.samples += 1
            write_yaml({'num_samples': self.samples}, f"{directory}/md.yml")

    def __len__(self):
        return self.samples

    def _get_worker_info(self):
        return torch.utils.data.get_worker_info() if self.distribute else None

    def _init_read_order(self):
        # Each node has the same worker_info, so the unique offsets for each is
        # rank * num_workers + worker_id
        # and the total available workers is world_size * num_workers
        worker_info = self._get_worker_info()
        files = sorted(list(glob.glob(f"{self.directory}/{self.pattern}")))

        if worker_info is None:
            num_workers_per_node = 1
            node_worker_id = 0
        else:
            num_workers_per_node = worker_info.num_workers
            node_worker_id = worker_info.id
        all_workers = (self.world_size * num_workers_per_node)
        offset = self.rank * num_workers_per_node + node_worker_id
        read_file_order = list(range(offset, len(files), all_workers))
        if not read_file_order:
            if offset > 0:
                # This is probably wrong
                logger.warning(f"There are no files to read for worker {node_worker_id}, offset {offset}!" +
                               " This might mean that you are passing an incorrect training or validation directory")
            else:
                # This is definitely wrong
                raise Exception(f"No files of pattern {self.pattern} were found in {self.directory}!")
        return files, read_file_order, node_worker_id

    def __iter__(self):
        files, read_file_order, _ = self._init_read_order()
        # If we have multiple files per worker, possibly shuffle the file read order
        while True:
            if self.shuffle:
                random.shuffle(read_file_order)
            for file_idx in read_file_order:
                file = files[file_idx]
                with open(file) as rf:
                    lines = rf.readlines()
                    if self.shuffle:
                        random.shuffle(lines)
                    for l in lines:
                        response = self.process_line(l)
                        if response:
                            yield response

    def process_line(self, line):
        """Read in a line and turn it into an entry

        The entries will get collated by the data loader

        :param line:
        :return:
        """


class NextTurnPredictionFileLoader(MultiFileLoader):

    def get_pair_order(self, pair):
        if self.record_keys == ['y', 'x']:
            return pair[1], pair[0]
        return pair[0], pair[1]

    def process_line(self, line):
        pair = line.strip().split('\t')
        # Unfortunately, this occassionally happens, a bunch of blank turns etc.
        if len(pair) != 2:
            return None
        q, r = self.get_pair_order(pair)
        if q == '' or r == '':
            return None
        if self.last_turn_only:
            turns = q.split('<EOU>')
            q = turns[-1] if turns[-1].strip() != '' else turns[-2]
            if q.strip() == '':
                return None
            q_vec, q_valid_lengths = self.src_vectorizer.run(q.split(), self.vocab)
        else:

            q = [self.src_vectorizer.vocab.get(x, Offsets.UNK) for x in self.src_vectorizer.iterable(q.split())]
            q_valid_lengths = len(q)
            if q_valid_lengths > self.src_vectorizer.mxlen:
                start = q_valid_lengths - self.src_vectorizer.mxlen
                q_vec = np.array(q[start:], dtype=np.long)
            else:
                q_vec = np.zeros(self.src_vectorizer.mxlen, dtype=np.long)
                q_vec[:q_valid_lengths] = np.array(q)

        r_vec, r_valid_lengths = self.tgt_vectorizer.run(r.split(), self.vocab)
        return q_vec, r_vec


def on_demand_mlm_masking(inputs, labels, mask_value, vocab_size):
    # Replace 15% of tokens
    masked_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).type(torch.bool)
    # Anything not masked is 0 so no loss
    labels[~masked_indices] = 0
    # Of the masked items, mask 80% of them with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.bool) & masked_indices
    inputs[indices_replaced] = mask_value
    # Replace 10% of them with random words, rest preserved for auto-encoding
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(
        torch.bool) & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=labels.device)
    inputs[indices_random] = random_words[indices_random]
    return inputs, labels, masked_indices


class SequencePredictionFileLoader(MultiFileLoader):

    def process_line(self, line):
        line = line.strip()
        if not line:
            return None

        vec, valid_lengths = self.src_vectorizer.run(line.split(), self.vocab)
        if valid_lengths < 2:
            return None
        return vec, vec


class PreprocessedFileLoader(MultiFileLoader):

    def process_line(self, line):
        obj = json.loads(line)
        if 'y' in obj.keys():
            return np.array(obj['x'], dtype=int), np.array(obj['y'], dtype=int)
        else:
            return np.array(obj['x'], dtype=int), np.array(obj['x'], dtype=int)


class NextSequencePredictionFileLoader(MultiFileLoader):

    def process_line(self, line):
        line = line.strip()
        if not line:
            return None
        vec, valid_lengths = self.src_vectorizer.run(line.split(), self.vocab)
        if valid_lengths < 2:
            return None
        pair_entry_length = self.src_vectorizer.mxlen//2
        end_of_query = min(valid_lengths//2, pair_entry_length)
        # Front half is all tokens up until the half_way marker
        # Create a new query vector
        query = np.zeros(pair_entry_length, dtype=np.int)
        query[:end_of_query] = vec[:end_of_query]
        # Repurpose the existing vector as the response vector
        vec = vec[end_of_query:end_of_query+pair_entry_length]
        return query, vec


class MultiTFRecordLoader(MultiFileLoader):
    """Using module tfrecord to read tfrecord file into PyTorch datasets"""
    try:
        import tfrecord
    except:
        pass
    def __init__(self, directory, vocabs, src_vectorizer, tgt_vectorizer, last_turn_only=True, distribute=True, shuffle=True, record_keys=None):
        super().__init__(directory, "*.tfrecord", vocabs, src_vectorizer, tgt_vectorizer, last_turn_only, distribute, shuffle)
        # create index first
        if not record_keys:
            self.x = 'x'
            self.y = 'y'
        elif len(record_keys) < 2:
            self.x = record_keys[0]
            self.y = record_keys[0]
        else:
            self.x, self.y = record_keys
        files = list(glob.glob(os.path.join(directory, '*.tfrecord')))
        for f in files:
            idx_file = '.'.join(f.split('.')[:-1]) + '.index'
            if not os.path.exists(idx_file):
                self.tfrecord.tools.tfrecord2idx.create_index(f, idx_file)

    def __iter__(self):
        files, read_file_order, node_worker_id = self._init_read_order()
        # If we have multiple files per worker, possibly shuffle the file read order
        while True:
            if self.shuffle:
                random.shuffle(read_file_order)
            for file_idx in read_file_order:
                file = files[file_idx]
                idx_file = '.'.join(file.split('.')[:-1]) + '.index'
                # shard = (worker_id, num_workers), but we already assigned this file to one certain worker,
                # so shard = (0, 1)
                itr = self.tfrecord.reader.tfrecord_loader(file, idx_file, shard=(0, 1))
                if self.shuffle:
                    np.random.seed(node_worker_id)
                    # not sure about the optimal choice of shuffle_queue_size here:
                    itr = self.tfrecord.iterator_utils.shuffle_iterator(itr, queue_size=128)
                for d in itr:
                    yield np.array(d[self.x], dtype=int), np.array(d[self.y], dtype=int)


class MultiFileDatasetReader:
    """Provide a base-class to do operations that are independent of token representation
    """

    def __init__(self, src_nctx=64, tgt_nctx=64, src_begin_tok=[], src_end_tok=['<EOS>'], tgt_begin_tok=['<GO>'],
                 tgt_end_tok=['<EOS>'], model_file=None, vocab_file=None, file_type='txt', reader_type="ntp",
                 record_keys=None, lower=False, extra_tokens=["[CLS]", "[MASK]"]):
        self.src_nctx = src_nctx
        self.tgt_nctx = tgt_nctx
        self.pattern = f'*.{file_type}'
        self.reader_type = reader_type
        if not src_begin_tok and self.reader_type == 'lang':
            src_begin_tok = ['[CLS]']
        self.record_keys = record_keys if record_keys else ['x', 'y']
        transform_fn = None if not lower else baseline.lowercase
        self.src_vectorizer = BPEVectorizer1D(model_file=model_file, vocab_file=vocab_file, mxlen=src_nctx,
                                              emit_begin_tok=src_begin_tok, emit_end_tok=src_end_tok,
                                              transform_fn=transform_fn, extra_tokens=extra_tokens)
        self.tgt_vectorizer = BPEVectorizer1D(model_file=model_file, vocab_file=vocab_file, mxlen=tgt_nctx,
                                              emit_begin_tok=tgt_begin_tok, emit_end_tok=tgt_end_tok,
                                              transform_fn=transform_fn, extra_tokens=extra_tokens)

    def build_vocab(self, _=None):
        return {'x': self.src_vectorizer.vocab}

    def load(self, directory, vocabs, distribute=True, shuffle=True):
        reader_type = self.reader_type.lower()
        # For `self.record_keys` in NTP and NSP, these names a
        if reader_type == "ntp":
            return NextTurnPredictionFileLoader(directory, self.pattern, vocabs, self.src_vectorizer, self.tgt_vectorizer, distribute=distribute, shuffle=shuffle, record_keys=self.record_keys)
        elif reader_type == "nsp":
            return NextSequencePredictionFileLoader(directory, self.pattern, vocabs, self.src_vectorizer, self.tgt_vectorizer, distribute=distribute, shuffle=shuffle)
        elif reader_type == "lang":
            print("Using files as an LM")
            return SequencePredictionFileLoader(directory, self.pattern, vocabs, self.src_vectorizer, self.tgt_vectorizer, distribute=distribute, shuffle=shuffle)
        elif reader_type == 'tfrecord':
            print("Reading data in .tfrecord format using the tfrecord module")
            return MultiTFRecordLoader(directory, vocabs, self.src_vectorizer, self.tgt_vectorizer, distribute=distribute, shuffle=shuffle, record_keys=self.record_keys)
        return PreprocessedFileLoader(directory, self.pattern, vocabs, self.src_vectorizer, self.tgt_vectorizer, distribute=distribute, shuffle=shuffle)


def get_lr_decay(sched_type, lr, steps_per_epoch, n_epochs, logger, decay_steps=None, decay_rate=None, alpha=None):
    if sched_type == 'cosine':
        decay_steps = decay_steps if decay_steps else steps_per_epoch * n_epochs
        alpha = alpha if alpha else 0.
        params = {'decay_steps': decay_steps, 'alpha': alpha}
    else:
        decay_steps = decay_steps if decay_steps else steps_per_epoch
        if not decay_rate:
            if sched_type == 'exponential':
                decay_rate = 0.5
            elif sched_type == 'invtime':
                decay_rate = 1.0
        params = {'decay_steps': decay_steps, 'decay_rate': decay_rate}
    lr_decay = create_lr_scheduler(lr_scheduler_type=sched_type, lr=lr, **params)
    logger.info(f"Using {sched_type} decay learning rate with params {params}.")
    return lr_decay

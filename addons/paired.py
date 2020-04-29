from torch.utils.data.dataset import IterableDataset, TensorDataset
from baseline.vectorizers import Token1DVectorizer, BPEVectorizer1D
from baseline.progress import create_progress_bar
from baseline.reader import register_reader
from eight_mile.utils import str2bool, write_yaml, read_yaml, Offsets
import torch
import glob
import numpy as np
import os


class MultiFileLoader(IterableDataset):

    def __init__(self, directory, pattern, vocabs, vectorizers, nctx):
        super().__init__()
        self.src_vectorizer = vectorizers['src']
        self.tgt_vectorizer = vectorizers['tgt']
        self.pattern = pattern
        self.nctx = nctx
        self.directory = directory
        self.vocab = vocabs
        self.samples = 0
        self.rank = 0
        self.world_size = 1
        if torch.distributed.is_initialized():
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

    def __iter__(self):
        # Each node has the same worker_info, so the unique offsets for each is
        # rank * num_workers + worker_id
        # and the total available workers is world_size * num_workers
        worker_info = torch.utils.data.get_worker_info()
        files = sorted(list(glob.glob(f"{self.directory}/{self.pattern}")))

        if worker_info is None:
            num_workers_per_node = 1
            node_worker_id = 0
        else:
            num_workers_per_node = worker_info.num_workers
            node_worker_id = worker_info.id
        all_workers = (self.world_size * num_workers_per_node)
        files_per_worker = len(files) // all_workers
        offset = self.rank * num_workers_per_node + node_worker_id
        start_idx = offset * files_per_worker
        end_idx = start_idx + files_per_worker if offset < all_workers - 1 else len(files)

        for file in files[start_idx:end_idx]:
            with open(file) as rf:
                for line in rf:
                    response = self.process_line(line)
                    if response:
                        yield response

    def process_line(self, line):
        """Read in a line and turn it into an entry

        The entries will get collated by the data loader

        :param line:
        :return:
        """

class Batcher(IterableDataset):

    def __init__(self, dataset, batchsz):
        self.batchsz = batchsz
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)//self.batchsz

    def _batch(self, batch_list):
        x = np.stack([item[0] for item in batch_list])
        y = np.stack([item[1] for item in batch_list])
        x_lens = np.stack([item[2] for item in batch_list])
        y_lens = np.stack([item[3] for item in batch_list])
        return {'src': x, 'tgt': y, 'src_lengths': x_lens, 'tgt_lengths': y_lens }

    def __iter__(self):

        dataset_iter = iter(self.dataset)
        steps_per_epoch = len(self.dataset)//self.batchsz
        for indices in range(steps_per_epoch):
            step = [next(dataset_iter) for _ in range(self.batchsz)]
            yield self._batch(step)


class NextTurnPredictionFileLoader(MultiFileLoader):

    def process_line(self, line):
        pair = line.strip().split('\t')
        # Unfortunately, this occassionally happens, a bunch of blank turns etc.
        if len(pair) != 2:
            return None
        q, r = pair
        q = q.strip()
        r = r.strip()
        if q == '' or r == '':
            return None
        q_vec, q_valid_lengths = self.src_vectorizer.run(reversed(q.split()), self.vocab)
        q_vec = np.roll(q_vec[::-1], -(self.src_vectorizer.mxlen - q_valid_lengths))
        r_vec, r_valid_lengths = self.tgt_vectorizer.run(['<GO>'] + r.split(), self.vocab)
        assert q_valid_lengths > 0 and q_valid_lengths <= self.src_vectorizer.mxlen
        assert r_valid_lengths > 0 and r_valid_lengths <= self.src_vectorizer.mxlen
        return q_vec, r_vec, q_valid_lengths, r_valid_lengths


@register_reader(task='seq2seq', name='paired-dir')
class MultiFileDatasetReader:
    """Provide a base-class to do operations that are independent of token representation
    """

    def __init__(self, vectorizers, trim, mxlen=64, pattern='*.txt', reader_type="ntp", **kwargs):
        self.nctx = mxlen
        self.pattern = pattern
        self.reader_type = reader_type
        self.vectorizers = vectorizers

    def build_vocabs(self, _=None, **kwargs):
        return {'src': self.vectorizers['src'].vocab}, self.vectorizers['tgt'].vocab

    def load(self, directory, src_vocabs, tgt_vocab, batch_size, **kwargs):
        # The src and target vectorizers are coupled
        loader = Batcher(NextTurnPredictionFileLoader(directory, self.pattern, tgt_vocab, self.vectorizers, self.nctx), batch_size)
        return loader


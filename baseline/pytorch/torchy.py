import math
import copy
import os
import logging
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from baseline.utils import lookup_sentence, get_version, Offsets
from eight_mile.pytorch.layers import *

PYT_MAJOR_VERSION = get_version(torch)

BaseLayer = nn.Module
TensorDef = torch.Tensor


class IterableDatasetAdapter(IterableDataset):
    def __init__(self, example_list, shuffle=False):
        super().__init__()
        self.examples = example_list
        self.shuffle = shuffle

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        while True:
            if self.shuffle:
                np.random.shuffle(self.examples)
            for example in self.examples:
                yield example

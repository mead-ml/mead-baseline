import math
import copy
import os
import logging
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from baseline.utils import lookup_sentence, get_version, Offsets
from eight_mile.pytorch.layers import *

PYT_MAJOR_VERSION = get_version(torch)

BaseLayer = nn.Module
TensorDef = torch.Tensor


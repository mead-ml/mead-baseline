"""Wav2vec2 pretraining using 8-mile API

"""
import logging
import time
import numpy as np
from typing import Tuple, List, Optional
import os
from argparse import ArgumentParser
import torch.nn as nn
import random
import soundfile as sf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset
from eight_mile.utils import str2bool, write_json, Average, get_num_gpus_multiworker
from eight_mile.optz import *
from eight_mile.pytorch.layers import save_checkpoint, init_distributed, Conv1DSame, TransformerEncoderStack, Dense, pytorch_conv1d
from eight_mile.pytorch.optz import *
from eight_mile.pytorch.serialize import convert_transformers_keys
import torch.nn.functional as F
from typing import Dict
logger = logging.getLogger(__file__)


CONV_FEATURES = {16: [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)],
                 8 : [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]}

START_TEMP = 2
END_TEMP = 0.5
TEMP_DECAY_FACTOR = 0.999995
XE_WGT = 0.1
DIVERSITY_WGT = 10


W2V2_FAIRSEQ_NESTED_LAYERS_MAP = {
    'encoder.layers.{}.self_attn.k_proj.weight':     'encoder.transformer.encoders.{}.self_attn.w_K.layer.weight',
    'encoder.layers.{}.self_attn.k_proj.bias':       'encoder.transformer.encoders.{}.self_attn.w_K.layer.bias',
    'encoder.layers.{}.self_attn.v_proj.weight':     'encoder.transformer.encoders.{}.self_attn.w_V.layer.weight',
    'encoder.layers.{}.self_attn.v_proj.bias':       'encoder.transformer.encoders.{}.self_attn.w_V.layer.bias',
    'encoder.layers.{}.self_attn.q_proj.weight':     'encoder.transformer.encoders.{}.self_attn.w_Q.layer.weight',
    'encoder.layers.{}.self_attn.q_proj.bias':       'encoder.transformer.encoders.{}.self_attn.w_Q.layer.bias',
    'encoder.layers.{}.self_attn.out_proj.weight':   'encoder.transformer.encoders.{}.self_attn.w_O.layer.weight',
    'encoder.layers.{}.self_attn.out_proj.bias':     'encoder.transformer.encoders.{}.self_attn.w_O.layer.bias',
    # Wav2vec2 ref impl is run with LN first
    'encoder.layers.{}.self_attn_layer_norm.weight': 'encoder.transformer.encoders.{}.ln1.weight',
    'encoder.layers.{}.self_attn_layer_norm.bias':   'encoder.transformer.encoders.{}.ln1.bias',
    'encoder.layers.{}.fc1.weight': 'encoder.transformer.encoders.{}.ffn.0.layer.weight',
    'encoder.layers.{}.fc1.bias':   'encoder.transformer.encoders.{}.ffn.0.layer.bias',
    'encoder.layers.{}.fc2.weight': 'encoder.transformer.encoders.{}.ffn.3.layer.weight',
    'encoder.layers.{}.fc2.bias':   'encoder.transformer.encoders.{}.ffn.3.layer.bias',
    'encoder.layers.{}.final_layer_norm.weight':  'encoder.transformer.encoders.{}.ln2.weight',
    'encoder.layers.{}.final_layer_norm.bias':   'encoder.transformer.encoders.{}.ln2.bias'

}

# We use a primitive from 8mi called Dense which owns the linear as a sub-layer, so convert those
W2V2_FAIRSEQ_FLAT_MAP = {
    'post_extract_proj.weight': 'proj_to_input.layer.weight',
    'post_extract_proj.bias': 'proj_to_input.layer.bias',
    'project_q.weight': 'project_q.layer.weight',
    'project_q.bias': 'project_q.layer.bias',
    'final_proj.weight': 'final_proj.layer.weight',
    'final_proj.bias': 'final_proj.layer.bias',
    'encoder.layer_norm.weight': 'encoder.transformer.ln.weight',
    'encoder.layer_norm.bias': 'encoder.transformer.ln.bias',
    'encoder.pos_conv.0.bias': 'encoder.pos_conv.conv.1.bias',
    'encoder.pos_conv.0.weight_g': 'encoder.pos_conv.conv.1.weight_g',
    'encoder.pos_conv.0.weight_v': 'encoder.pos_conv.conv.1.weight_v',
    #'layer_norm.weight': 'encoder.ln.weight',
    #'layer_norm.bias': 'encoder.ln.bias'
}



def convert_keys(num_layers: int, d: Dict, nested_layer_map: Dict = W2V2_FAIRSEQ_NESTED_LAYERS_MAP, flat_map: Dict = W2V2_FAIRSEQ_FLAT_MAP) -> Dict:

    m = {}
    for i in range(num_layers):
        for k, v in nested_layer_map.items():
            key = k.format(i)
            m[v.format(i)] = d.pop(key)

    for k, v in flat_map.items():
        m[v] = d.pop(k)

    for k, v in d.items():
        m[k] = v



    return m

def load_fairseq_bin(w2v: nn.Module, bin_file: str, nested_layer_map=W2V2_FAIRSEQ_NESTED_LAYERS_MAP, flat_map=W2V2_FAIRSEQ_FLAT_MAP):

    d = torch.load(bin_file)["model"]
    transformer = w2v.encoder.transformer
    num_layers = len(transformer.encoders)
    mapped_keys = convert_keys(num_layers, d, nested_layer_map, flat_map)
    unknown_keys = w2v.load_state_dict(mapped_keys, strict=False)
    missing_keys = [key for key in unknown_keys.missing_keys]
    return {'missing': missing_keys, 'unexpected': unknown_keys.unexpected_keys}


class AudioFileDataset(IterableDataset):

    def __init__(self, manifest, max_length, target_tokens_per_batch, distribute=True, shuffle=True,  min_length=0):
        super().__init__()
        self.max_length = max_length
        self.manifest = manifest
        self.rank = 0
        self.world_size = 1
        self.files = []
        self.target_tokens_per_batch = target_tokens_per_batch
        self.shuffle = shuffle
        self.distribute = distribute
        if torch.distributed.is_initialized() and distribute:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

        self._read_manifest(manifest, min_length)

    def _read_manifest(self, manifest, min_length):
        skipped = 0
        with open(manifest, "r") as f:
            self.directory = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                sz = int(items[1])
                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.files.append((os.path.join(self.directory, items[0]), sz,))

        sorted(self.files, key=lambda item: item[-1])
        logger.info(f"loaded {len(self.files)}, skipped {skipped} samples")

    def _get_worker_info(self):
        return torch.utils.data.get_worker_info() if self.distribute else None

    def _init_read_order(self):
        # Each node has the same worker_info, so the unique offsets for each is
        # rank * num_workers + worker_id
        # and the total available workers is world_size * num_workers
        worker_info = self._get_worker_info()

        if worker_info is None:
            num_workers_per_node = 1
            node_worker_id = 0
        else:
            num_workers_per_node = worker_info.num_workers
            node_worker_id = worker_info.id
        all_workers = (self.world_size * num_workers_per_node)
        offset = self.rank * num_workers_per_node + node_worker_id
        read_file_order = list(range(offset, len(self.files), all_workers))
        if not read_file_order:
            if offset > 0:
                # This is probably wrong
                logger.warning(f"There are no files to read for worker {node_worker_id}, offset {offset}!" +
                               " This might mean that you are passing an incorrect training or validation directory")
            else:
                # This is definitely wrong
                raise Exception(f"No files of pattern {self.pattern} were found in {self.directory}!")
        return read_file_order, node_worker_id

    def next_sample(self):
        read_file_order, _ = self._init_read_order()
        # If we have multiple files per worker, possibly shuffle the file read order
        while True:
            if self.shuffle:
                random.shuffle(read_file_order)
            for file_idx in read_file_order:
                file, _ = self.files[file_idx]
                yield self.process_sample(file, self.max_length)

    def process_sample(self, file, len):
        """Read in a line and turn it into an entry.  FIXME, get from anywhere

        The entries will get collated by the data loader

        :param file:
        :return:
        """
        wav, _ = sf.read(file)
        wav = wav.astype(np.float32)
        return wav[:len]

    def __iter__(self):

        min_length = self.max_length

        num_tokens_predicted = 0

        samples = []
        for sample in self.next_sample():

            if num_tokens_predicted < self.target_tokens_per_batch:
                min_length = min(min_length, len(sample))
                samples.append(sample)
                num_tokens_predicted = len(samples) * min_length
            else:
                batch = np.stack([s[:min_length] for s in samples])
                samples = []
                num_tokens_predicted = 0
                #logger.debug("(%d, %d) %d", batch.shape[0], batch.shape[1], np.product(batch.shape))
                yield batch


def find_fit(v, fits):
    truncate_to = 0
    for fit in fits:
        if v//fit:
            truncate_to = fit
        else:
            break
    return truncate_to


class BucketingAudioDataset(AudioFileDataset):

    def __init__(self, buckets, manifest, max_length, target_tokens_per_batch, distribute=True, shuffle=True,  min_length=0):
        self.bucket_lengths = buckets
        super().__init__(manifest, max_length, target_tokens_per_batch, distribute, shuffle, min_length)

    def _read_manifest(self, manifest, _):
        skipped = 0
        asc = sorted(self.bucket_lengths)
        self.files = {b: [] for b in asc}

        num_samples = 0
        with open(manifest, "r") as f:

            directory = f.readline().strip()
            for line in f:
                num_samples += 1
                items = line.strip().split("\t")
                sz = int(items[1])
                fname = os.path.join(directory, items[0])

                if sz < asc[0]:
                    skipped += 1
                    continue
                count = find_fit(sz, self.bucket_lengths)
                self.files[count].append((fname, sz))

        logger.info('Num samples %d, skipped %d', num_samples, skipped)

    def next_sample(self):
        read_file_order, _ = self._init_read_order()
        keys = list(self.files.keys())

        # If we have multiple files per worker, possibly shuffle the file read order
        while True:
            if self.shuffle:
                random.shuffle(read_file_order)
            for bucket_idx in read_file_order:
                bucket = keys[bucket_idx]
                for (file, _) in self.files[bucket]:
                    yield self.process_sample(file, bucket)


def timestep_masking(
        shape: Tuple[int, int],
        p_start: float = 0.65,
        mask_length: int = 10
) -> np.ndarray:
    bsz, input_length = shape
    mask = np.full((bsz, input_length), False)
    num_mask = int(p_start * input_length / float(mask_length) + np.random.rand())
    mask_idcs = []
    for i in range(bsz):
        sz = input_length
        lengths = np.full(num_mask, mask_length)
        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

        mask_idc = np.asarray(
            [
                mask_idc[j] + offset
                for j in range(len(mask_idc))
                for offset in range(lengths[j])
            ]
        )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    ls = [len(m) for m in mask_idcs]
    min_len = min(ls)
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


class Loss(nn.Module):
    def __init__(self, n_vars, n_negatives=100):
        super().__init__()
        self.n_vars = n_vars
        self.sample = Sampler(n_negatives)

    def __call__(self, model, features):
        outputs, latents, gs_probs, time_mask = model(features)
        y = latents.unsqueeze(0)
        outputs_shape = outputs.shape
        outputs = outputs[time_mask.unsqueeze(-1).expand_as(outputs)].view(outputs_shape[0], -1, outputs_shape[-1])
        outputs = outputs.unsqueeze(0)
        neg, _ = self.sample.negatives(latents)
        targets = torch.cat([y, neg], dim=0)
        logits = torch.cosine_similarity(outputs, targets, dim=-1)
        logits = logits.transpose(2, 0)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        diversity = DIVERSITY_WGT * (self.n_vars - gs_probs) / self.n_vars
        xe_loss = F.cross_entropy(logits, targets)
        cross_entropy = XE_WGT * xe_loss
        return cross_entropy + diversity


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()


        def block(
            n_in,
            n_out,
            k,
            stride,
            is_group_norm=False,
            conv_bias=False,
        ):

            if is_group_norm:
                return nn.Sequential(
                    pytorch_conv1d(n_in, n_out, k, initializer="kaiming", stride=stride, bias=conv_bias),
                    nn.Dropout(p=dropout),
                    nn.GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(pytorch_conv1d(n_in, n_out, k, initializer="kaiming", stride=stride, bias=conv_bias),
                                     nn.Dropout(p=dropout),
                                     nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_group_norm=i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)
        # BxCxT -> BxTxC
        x = x.transpose(1, 2)
        return x


class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        min_temperature,
        max_temperature,
        temperature_decay,
        num_groups,
        vq_dim
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temperature: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            vq_dim: dimensionality of the resulting quantized vector
        """
        super().__init__()

        self.num_groups = num_groups
        self.input_dim = dim
        self.num_vars = num_vars

        assert (
            vq_dim % num_groups == 0
        ), f"dim {vq_dim} must be divisible by groups {num_groups} for concatenation"

        # per var
        var_dim = vq_dim // num_groups
        # vars count is the groups by the number of vars per group
        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        # projection
        self.weight_proj = nn.Linear(self.input_dim, num_groups * num_vars)
        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.curr_temperature = self.max_temperature
        # Why dont they init this, I guess because its not necessarily used in training
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temperature = max(
            self.max_temperature * self.temperature_decay ** num_updates, self.min_temperature
        )

    # Create codebook on the fly
    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.num_groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            self.codebook_indices = self.codebook_indices.view(
                self.num_vars ** self.num_groups, -1
            )
            for b in range(1, self.num_groups):
                self.codebook_indices[:, b] += self.num_vars * b
            self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
                .index_select(0, indices)
                .view(self.num_vars ** self.num_groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.num_groups)
        cb_size = indices.size(0)
        assert (
                n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.num_groups):
            exponent = self.num_groups - i - 1
            res += indices[..., i] * (self.num_vars ** exponent)
        return res

    def targets_for(self, x):
        """Get the output of the gumbel softmax or hard estimator and convert to one-hots

        :param x: [B, T, GxV]
        :return: y [B, T, G]
        """
        bsz = x.shape[0]
        tsz = x.shape[1]
        x = x.view(bsz * tsz, -1)
        targets = x.view(bsz * tsz * self.num_groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
        return targets

    def forward(self, x):

        bsz, tsz, fsz = x.shape
        # This should NOT be required, PyTorch folds under the hood
        x = self.weight_proj(x)
        # The output back out is BxTx(GxV)
        x = x.view(bsz * tsz * self.num_groups, -1)
        avg_probs = torch.softmax(
            x.float(), dim=-1
        ).mean(dim=0)

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temperature, hard=True).type_as(x)
        else:
            # Max over vars
            _, k = x.max(-1)
            hard_x = (
                x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.num_groups, -1)
            )
            x = hard_x

        x = x.view(bsz * tsz, self.num_groups, -1)
        prob_ppl = torch.sum(torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), -1)
        ))

        # broadcast the quantization table
        # [B, T, (GxV), 1] *. [1, (GxV), qsz] = [B, T, (GxV), qsz]
        x = x.view(bsz * tsz, -1, 1)
        x = x * self.vars

        x = x.view(bsz * tsz, self.num_groups, self.num_vars, -1)
        # This collapses over the variables
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)
        return x, prob_ppl


class AudioTransformerEncoder(nn.Module):
    def __init__(
            self,
            num_heads: int,
            d_model: int,
            pdrop: float,
            layers: int = 1,
            activation: str = "gelu",
            d_ff: Optional[int] = None,
            **kwargs):
        super().__init__()
        self.d_model = d_model
        self.conv_pos_kernel = kwargs.get('conv_pos_kernel', 128)
        self.conv_groups = kwargs.get('conv_groups', 16)
        self.dropout = nn.Dropout(pdrop)

        std = math.sqrt((4 * (1.0 - pdrop)) / (self.conv_pos_kernel * self.d_model))
        self.pos_conv = Conv1DSame(d_model, d_model, self.conv_pos_kernel, activation="gelu", groups=self.conv_groups, unif=std, initializer="normal")
        self.pos_conv.conv[1] = nn.utils.weight_norm(self.pos_conv.conv[1], name="weight", dim=2)
        if not d_ff:
            d_ff = 4*d_model

        self.transformer = TransformerEncoderStack(num_heads=num_heads,
                                                   d_model=d_model,
                                                   pdrop=pdrop,
                                                   layers=layers,
                                                   activation=activation,
                                                   layer_norms_after=False,
                                                   d_ff=d_ff)
        #self.ln = nn.LayerNorm(self.d_model)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)
        #x = self.ln(x)
        return x

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)

        #x_conv = self.pos_conv(x)
        x += x_conv
        x = self.dropout(x)
        self.transformer((x, padding_mask))

        return x



class Wav2Vec2Model(nn.Module):
    def __init__(self, conv_features, num_vq_vars, start_temp, end_temp, temp_decay_factor,
                 num_vq_groups, d_model, num_heads, num_layers, dropout=0.1, d_ff=None, final_dim=256, dropout_input=0.1, dropout_features=0.1):
        super().__init__()
        fx_dsz = conv_features[-1][0]
        self.layer_norm = torch.nn.LayerNorm(fx_dsz)
        self.dropout_input = torch.nn.Dropout(dropout_input)
        self.dropout_features = torch.nn.Dropout(dropout_features)

        self.feature_extractor = ConvFeatureExtractionModel(conv_features)
        self.proj_to_input = Dense(fx_dsz, d_model)
        self.quantizer = GumbelVectorQuantizer(fx_dsz, num_vq_vars, start_temp, end_temp, temp_decay_factor, num_vq_groups, final_dim)
        self.encoder = AudioTransformerEncoder(num_heads, d_model, dropout, num_layers, d_ff=d_ff)
        self.project_q = Dense(final_dim, final_dim)
        self.final_proj = Dense(d_model, final_dim)
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(d_model).uniform_()
        )

    def set_num_updates(self, s):
        self.quantizer.set_num_updates(s)

    def forward(self, x):

        fx = self.feature_extractor(x)
        features = self.layer_norm(fx)
        unmasked_features = features.clone()
        features = self.proj_to_input(features)
        B, T, _ = unmasked_features.shape
        features = self.dropout_input(features)
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        time_mask = timestep_masking((B, T))
        time_mask = torch.from_numpy(time_mask).to(x.device)
        features[time_mask] = self.mask_emb

        y = unmasked_features[time_mask].view(
            unmasked_features.size(0), -1, unmasked_features.size(-1)
        )
        x = self.encoder(features)
        y, vq_probs = self.quantizer(y)

        y = self.project_q(y)
        x = self.final_proj(x)
        return x, y, vq_probs, time_mask


class Sampler:

    def __init__(self, n_negatives=100):
        self.n_negatives = n_negatives

    def negatives(self, y):

        B, T, C = y.shape
        y = y.view(-1, C)  # BTC => (BxT)C

        with torch.no_grad():
            Ts = torch.arange(T).unsqueeze(-1)
            Ts = Ts.expand(-1, self.n_negatives)
            Ts = Ts.reshape(-1)
            neg_idxs = np.random.randint(0, T-1, (B, self.n_negatives*T))
            neg_idxs = torch.from_numpy(neg_idxs)
            neg_idxs[neg_idxs >= Ts] += 1
            stride = torch.arange(B) * T
            stride = stride.unsqueeze(-1)

        neg_idxs = neg_idxs + stride
        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            B, T, self.n_negatives, C
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--train_manifest_file", type=str, default="train.tsv", help='File path to use for train file')
    parser.add_argument("--valid_manifest_file", type=str, default="valid.tsv", help='File path to use for valid file')
    parser.add_argument("--dataset_key", default="ls",
                        help="dataset key for basedir")
    parser.add_argument("--num_vq_vars", type=int, default=320)
    parser.add_argument("--num_vq_groups", type=int, default=2)
    parser.add_argument("--sr", type=int, choices=[8, 16], default=16)
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=3072, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--tokens_per_batch", type=int, default=1_400_000, help="Number of tokens per batch")
    parser.add_argument("--max_sample_len", type=int, default=250_000, help="Max sample length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="The type of learning rate decay scheduler")
    parser.add_argument("--lr_decay_steps", type=int, help="decay steps of lr scheduler")
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of lr scheduler")
    parser.add_argument("--lr_alpha", type=float, default=0., help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--optim", default="adamw", type=str, help="Optimizer to use (defaults to adamw)")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--bucketing", type=str2bool, default=False, help="Bucket the inputs to fixed batch sizes?")
    parser.add_argument("--buckets", type=int, nargs="+",
                        help="Bucket sizes if bucketing",
                        default=[11111, 35714, 38461, 41666, 45454, 50000, 55555, 62500, 71428, 83333, 100000, 125000, 166666, 250000])

    parser.add_argument("--train_steps", type=int, default=400_000, help="Num training steps")
    parser.add_argument("--valid_steps", type=int, default=10_000, help="Num valid steps to evaluate each time")

    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--steps_per_checkpoint", type=int, default=1000, help="The number of steps per checkpoint")
    parser.add_argument("--preprocessed", type=str2bool, default=True, help="Has the data already been preprocessed?")
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

    if args.basedir is None:
        args.basedir = 'wav2vec2-{}-{}'.format(args.dataset_key, os.getpid())
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    os.makedirs(args.basedir, exist_ok=True)
    num_gpus = get_num_gpus_multiworker()
    args.distributed = args.distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")

    if args.distributed:
        args.device, updated_local_rank = init_distributed(args.local_rank)
        args.local_rank = updated_local_rank

    train_manifest = os.path.join(args.manifest_dir, args.train_manifest_file)
    valid_manifest = os.path.join(args.manifest_dir, args.valid_manifest_file)
    if args.bucketing:
        train_set = BucketingAudioDataset(args.buckets, train_manifest, args.max_sample_len, args.tokens_per_batch)
        valid_set = BucketingAudioDataset(args.buckets, valid_manifest, args.max_sample_len, args.tokens_per_batch)
    else:
        train_set = AudioFileDataset(train_manifest, args.max_sample_len, args.tokens_per_batch)
        valid_set = AudioFileDataset(valid_manifest, args.max_sample_len, args.tokens_per_batch)
    train_loader = DataLoader(train_set, batch_size=None, num_workers=args.num_train_workers)
    valid_loader = DataLoader(valid_set, batch_size=None)
    logger.info("Loaded datasets")

    model = Wav2Vec2Model(CONV_FEATURES[args.sr], args.num_vq_vars,
                          START_TEMP, END_TEMP, TEMP_DECAY_FACTOR, args.num_vq_groups, args.d_model,
                          args.num_heads, args.num_layers,
                          args.dropout, args.d_ff).cuda()

    loss_function = Loss(args.num_vq_vars * args.num_vq_groups).to(args.device)
    logger.info("Loaded model and loss")

    # according to pytorch, len(train_loader) will return len(train_set) when train_set is IterableDataset, so manually
    # correct it here
    valid_steps = args.valid_steps
    update_on = args.steps_per_checkpoint
    validate_on = update_on * 10
    report_on = max(10, update_on) // 10
    lr_decay = CosineDecaySchedulerPyTorch(decay_steps=args.train_steps, alpha=args.lr_alpha, lr=args.lr)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, lr_decay, lr=args.lr)

    global_step = 0
    if args.restart_from:

        if args.restart_from.endswith('.pt'):
            print(load_fairseq_bin(model, args.restart_from))
        else:
            model.load_state_dict(torch.load(args.restart_from))
            vec = args.restart_from.split("-")
            global_step = int(vec[-1].split(".")[0])
            logger.info("Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d",
                        args.restart_from, global_step)

    optimizer = OptimizerManager(model, global_step, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay)
    logger.info("Model has {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Prepare model for distributed training if needed
    if args.distributed:
        # This program assume pure data parallelism, each model is on a single gpu
        # If we wanted to support model and data parallelism we would need to update
        # the selection of gpus based on rank, it would need to select multiple ids
        # based on rank, here we select only a single gpu and use it for input and
        # output.
        model = DistributedDataParallel(model, device_ids=[args.device], output_device=args.device)
        logger.info("Model located on %s", args.device)

    model_base = os.path.join(args.basedir, 'checkpoint')
    steps = global_step

    train_itr = iter(train_loader)
    start_of_run = 0
    avg_loss = Average('average_train_loss')
    step_time = Average('average_step_time')
    for i in range(steps, args.train_steps):

        metrics = {}
        optimizer.zero_grad()
        start = time.time()
        model.train()
        # This loader will iterate for ever
        batch = next(train_itr)
        steps += 1
        inputs = batch.to(args.device)
        loss = loss_function(model, inputs)
        loss.backward()
        avg_loss.update(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        optimizer.zero_grad()
        elapsed = time.time() - start
        step_time.update(elapsed)

        if (steps + 1) % report_on == 0:
            steps_per_sec = 1.0 / step_time.avg
            logging.info('%s, steps/min %f, LR %.6f', avg_loss, steps_per_sec*60, optimizer.current_lr)

        if (steps + 1) % update_on == 0 and args.local_rank < 1:
            save_checkpoint(model, model_base, steps, tick_type='step')
        if (steps + 1) % validate_on == 0 and args.local_rank < 1:
            # How much time elapsed in minutes
            elapsed = (time.time() - start_of_run) / 60
            metrics['train_elapsed_min'] = elapsed

            train_token_loss = avg_loss.avg
            metrics['average_train_loss'] = train_token_loss
            avg_valid_loss = Average('average_valid_loss')

            model.eval()
            valid_start = time.time()
            valid_itr = iter(valid_loader)
            for j in range(valid_steps):
                batch = next(valid_itr)
                with torch.no_grad():
                    x = batch.to(args.device)
                    loss = loss_function(model, x)
                    avg_valid_loss.update(loss.item())
            valid_token_loss = avg_valid_loss.avg
            metrics['average_valid_loss'] = valid_token_loss
            elapsed = (time.time() - valid_start) / 60
            metrics['valid_elapsed_epoch'] = elapsed
            logger.info(metrics)


if __name__ == "__main__":
    train()

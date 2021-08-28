import os
import argparse
import torch
from typing import Tuple, Dict
from eight_mile.pytorch.layers import EmbeddingsStack
from eight_mile.pytorch.serialize import *
from baseline.pytorch.lm import TransformerMaskedLanguageModel
from eight_mile.utils import read_config_stream

from eight_mile.pytorch.embeddings import LookupTableEmbeddings, LearnedPositionalLookupTableEmbeddings
from eight_mile.downloads import web_downloader
# From https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py

"""

You can use the predefined checkpoint paths, or you can download the model and convert
the checkpoint that way using the git LFS repos provided by hugging face.  This is 
the preferred approach for most models.  For any model supported, you can go to its page on
https://huggingface.co/ and click on the `Use in transformers` link to get the git LFS repo

For example to convert SentenceBERT, which is just a vanilla BERT style checkpoint:

git clone https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
and then pass that path in



"""
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-config.json",
    "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-config.json",
    "bert-base-japanese": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-config.json",
    "bert-base-japanese-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-config.json",
    "bert-base-japanese-char": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-config.json",
    "bert-base-japanese-char-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-config.json",
    "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/config.json",
    "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/config.json",
    "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/config.json",
}

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    "bert-base-japanese": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-pytorch_model.bin",
    "bert-base-japanese-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-pytorch_model.bin",
    "bert-base-japanese-char": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-pytorch_model.bin",
    "bert-base-japanese-char-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
    "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/pytorch_model.bin",
}

MODEL_MAPS = {
    'roberta': {'layers': ROBERTA_HF_LAYER_MAP, 'embed': ROBERTA_HF_EMBED_MAP},
    'bert': {'layers': BERT_HF_LAYER_MAP, 'embed': BERT_HF_EMBED_MAP}
}


def create_transformer_lm(config_url: str) -> Tuple[TransformerMaskedLanguageModel, int]:
    config = read_config_stream(config_url)
    pdrop = config['attention_probs_dropout_prob']
    activation = config['hidden_act']
    d_model = config['hidden_size']
    d_ff = config['intermediate_size']
    layer_norm_eps = float(config['layer_norm_eps'])
    mxlen = config['max_position_embeddings']
    num_heads = config['num_attention_heads']
    num_layers = config['num_hidden_layers']
    pad = config['pad_token_id']
    if pad != 0 and pad != 1:
        raise Exception(f"Unexpected pad value {pad}")
    tt_vsz = config['type_vocab_size']
    vsz = config['vocab_size']
    embeddings = {'x': LearnedPositionalLookupTableEmbeddings(vsz=vsz, dsz=d_model, mxlen=mxlen),
                  'tt': LookupTableEmbeddings(vsz=tt_vsz, dsz=d_model)}
    model = TransformerMaskedLanguageModel.create(embeddings,
                                                  d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                                                  tgt_key='x',
                                                  num_layers=num_layers,
                                                  embeddings_dropout=pdrop,
                                                  dropout=pdrop,
                                                  activation=activation,
                                                  layer_norms_after=True,
                                                  embeddings_reduction='sum-layer-norm')
    return model, num_layers


def convert_checkpoint(bert_checkpoint: str, num_layers: int, target_dir: str, checkpoint_disk_loc: str,
                       nested_layer_map, flat_map, model_type) -> Dict:

    if os.path.exists(checkpoint_disk_loc):
        print(f'Checkpoint found at {checkpoint_disk_loc}')
    else:
        print(f'Downloading {bert_checkpoint} to {checkpoint_disk_loc}')
        web_downloader(bert_checkpoint, checkpoint_disk_loc)
    state_dict = torch.load(checkpoint_disk_loc)
    if model_type:
        keys = list(state_dict.keys())
        for k in keys:
            v = state_dict[k]
            if not k.startswith(model_type):
                state_dict[f'{model_type}.{k}'] = v
                del state_dict[k]

    mapped_keys = convert_transformers_keys(num_layers, state_dict, nested_layer_map=nested_layer_map, flat_map=flat_map)
    return mapped_keys

def write_npz(output_file: str, model: TransformerMaskedLanguageModel):
    save_tlm_npz(model, output_file)


parser = argparse.ArgumentParser(description='Grab a HuggingFace BERT checkpoint down and convert it to a TLM NPZ file')
parser.add_argument('--model', help='This is the key of a HuggingFace input model or path to model', default='bert-base-uncased')
parser.add_argument('--model_type', choices=['bert', 'roberta'], default='Model flavor: bert (BERT, SBERT), roberta (RoBERTa, XLM-R)')
parser.add_argument('--target_dir', help='This is the target directory where we will put the checkpoints')
parser.add_argument('--config_file_name', help='The name of the config file.  Only needed for local models', default='config.json')
parser.add_argument('--checkpoint', help='The name of the checkpoint file. Only needed for local models', default='pytorch_model.bin')
args = parser.parse_args()

if os.path.isdir(args.model):
    config_url = os.path.join(args.model, args.config_file_name)
    pt_checkpoint = os.path.join(args.model, args.checkpoint)
    checkpoint_disk_loc = pt_checkpoint
    if not args.target_dir:
        args.target_dir = args.model
    output_file = os.path.basename(args.model)
else:
    config_url = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP[args.model]
    pt_checkpoint = BERT_PRETRAINED_MODEL_ARCHIVE_MAP[args.model]
    checkpoint_basename = os.path.basename(pt_checkpoint)
    checkpoint_disk_loc = os.path.join(args.target_dir, checkpoint_basename)
    output_file = args.model

model, num_layers = create_transformer_lm(config_url)
mapped_keys = convert_checkpoint(pt_checkpoint, num_layers, args.target_dir, checkpoint_disk_loc,
                                 nested_layer_map=MODEL_MAPS[args.model_type]['layers'],
                                 flat_map=MODEL_MAPS[args.model_type]['embed'], model_type=args.model_type)
unknown_keys = model.load_state_dict(mapped_keys, strict=False)
for k in unknown_keys.missing_keys:
    if k not in ['output_layer.weight', 'output_layer.bias']:
        print(f'Warning: missing key: {k}')
for k in unknown_keys.unexpected_keys:
    print(f'Warning: unexpected key {k}')
output_file = os.path.join(args.target_dir, output_file + '.npz')
print(f'Writing output file {output_file}')
write_npz(output_file, model)

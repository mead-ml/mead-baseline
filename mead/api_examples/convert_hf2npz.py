import os
import argparse
import torch
from typing import Tuple, Dict
from eight_mile.pytorch.layers import EmbeddingsStack
from eight_mile.pytorch.serialize import *
from baseline.pytorch.lm import TransformerMaskedLanguageModel, TransformerLanguageModel
from eight_mile.utils import read_config_stream

from eight_mile.pytorch.embeddings import LookupTableEmbeddings, LearnedPositionalLookupTableEmbeddings
from eight_mile.downloads import web_downloader

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
    'bert': {'layers': BERT_HF_LAYER_MAP, 'embed': BERT_HF_EMBED_MAP},
    'gpt2': {'layers': GPT2_HF_LAYER_MAP, 'embed': GPT2_HF_EMBED_MAP}

}


def create_transformer_lm(config_url: str, model_type: str) -> Tuple[TransformerMaskedLanguageModel, int]:
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

    if model_type == "bert" or model_type == "roberta":
        embeddings_type = "sum-layer-norm"
        transformer_type = "post-layer-norm"
    else:
        raise Exception(f"We dont support model type {model_type}")

    embeddings = {'x': LearnedPositionalLookupTableEmbeddings(vsz=vsz, dsz=d_model, mxlen=mxlen), 'tt': LookupTableEmbeddings(vsz=tt_vsz, dsz=d_model)}
    if model_type == "bert":
        embeddings['tt']: LookupTableEmbeddings(vsz=tt_vsz, dsz=d_model)

    model = TransformerMaskedLanguageModel.create(embeddings,
                                                  d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                                                  tgt_key='x',
                                                  num_layers=num_layers,
                                                  embeddings_dropout=pdrop,
                                                  dropout=pdrop,
                                                  activation=activation,
                                                  transformer_type=transformer_type,
                                                  embeddings_reduction=embeddings_type)
    return model, num_layers


def create_transformer_lm_gpt2(config_url: str, model_type: str) -> Tuple[TransformerLanguageModel, int]:
    config = read_config_stream(config_url)
    pdrop = config['attn_pdrop']
    activation = config['activation_function']
    d_model = config['n_embd']
    d_ff = 4 * d_model
    layer_norm_eps = float(config['layer_norm_epsilon'])
    mxlen = config['n_ctx']
    num_heads = config['n_head']
    num_layers = config['n_layer']
    pad = 0
    if pad != 0 and pad != 1:
        raise Exception(f"Unexpected pad value {pad}")
    vsz = config['vocab_size']
    embeddings_type = "sum"
    transformer_type = "pre-layer-norm"
    embeddings = {'x': LearnedPositionalLookupTableEmbeddings(vsz=vsz, dsz=d_model, mxlen=mxlen)}
    model = TransformerLanguageModel.create(embeddings,
                                                  d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                                                  tgt_key='x',
                                                  num_layers=num_layers,
                                                  embeddings_dropout=pdrop,
                                                  dropout=pdrop,
                                                  activation=activation,
                                                  transformer_type=transformer_type,
                                                  embeddings_reduction=embeddings_type, layer_norms_after=False, layer_norm_eps= layer_norm_eps, tie_weights=True)
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
    if model_type == 'gpt2':
        mapped_keys = convert_transformers_keys_gpt2(num_layers, state_dict, nested_layer_map=nested_layer_map, flat_map=flat_map)
    else:
        mapped_keys = convert_transformers_keys(num_layers, state_dict, nested_layer_map=nested_layer_map,
                                                     flat_map=flat_map)
    return mapped_keys

def write_npz(output_file: str, model: TransformerMaskedLanguageModel):
    save_tlm_npz(model, output_file)


parser = argparse.ArgumentParser(description='Grab a HuggingFace BERT checkpoint down and convert it to a TLM NPZ file')
parser.add_argument('--model', help='This is the key of a HuggingFace input model or path to model', default='bert-base-uncased')
parser.add_argument('--model_type', choices=['bert', 'roberta', 'gpt2'], help='Model flavor: bert (BERT, SentenceBERT), roberta (RoBERTa, XLM-R, CamemBERT), gpt2')
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
if args.model_type == 'gpt2':
    model, num_layers = create_transformer_lm_gpt2(config_url, args.model_type)
else:
    model, num_layers = create_transformer_lm(config_url, args.model_type)
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
def compare_gpt2(model: torch.nn.Module, gpt_model_path: str,  query_list: str, max_to_complete: int =20):
    model.eval()
    model.to("cuda")
    import re

    #loading Hugging Face GPT2LM head
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    hf_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path)
    hf_model = GPT2LMHeadModel.from_pretrained(gpt_model_path)
    hf_model.eval()

    ## running both the models across query list
    with torch.no_grad():
        for query in query_list:
            print(f"\t\tQuery: {query}")
            outputs = []
            for i in range(max_to_complete):
                encoded_input = hf_tokenizer(query, return_tensors='pt')
                ids = encoded_input['input_ids'][0].to('cuda')

                #npz mead model output
                response = model({'x': ids.unsqueeze(0)}, None)[0].squeeze(0)
                response = response[len(ids) - 1]
                response = response.argmax(-1).item()

                # gpt model model output
                output_hf_model = hf_model(**encoded_input)
                hf_response = output_hf_model[0][0][-1].argmax(-1).item()
                assert(hf_response == response), f'Prediction Mismatch Error:\
                 Without any sampling Hugging Face model predicted the subword: {hf_tokenizer.decode([hf_response])}, where as\
                 Mead Model predicted the subword: \
                 {hf_tokenizer.decode([response])}, for the query text: {query}'
                outputs.append(response)
                query = f'{query}{hf_tokenizer.decode([response])}'
            outputs = hf_tokenizer.decode(outputs)
            outputs = re.sub('\s+',' ',outputs.strip())
            print(f"\t\tGenerated Sequence:{outputs}\n\n")
    return outputs

if args.model_type == 'gpt2':
     query_list = ['in the great green room there was a telephone and a red balloon and a picture of - the cow jumping over the moon.',\
     'hello how are you', "Hello, I'm a language model,", "What can i do for you?", "Replace me by any text you'd like"]
     compare_gpt2(model, args.model, query_list)

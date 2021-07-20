"""Convert a fine-tuned Transformer model from Baseline PyTorcy output to NPZ

These models will have the entire transformer inside the primary embeddings key,
and a single output layer on the model.  We will use
`eight_mile.pytorch.serialize.save_tlm_output_npz()`

Because these models are serialized with pickle, if you used an addon, you need to
ensure that this addon is in the path.  This can be done via the command-line by
passing `--modules m1 m2 ...`
"""

import argparse
import os
import torch
import logging
from baseline.utils import import_user_module
from eight_mile.pytorch.serialize import save_tlm_output_npz, save_tlm_npz

def main():
    parser = argparse.ArgumentParser(
        description='Convert finetuned transformer model trained with PyTorch classifier to an TLM NPZ'
    )
    parser.add_argument('--model', help='The path to the .pyt file created by training', required=True, type=str)
    parser.add_argument('--device', help='device')
    parser.add_argument('--npz', help='Output file name, defaults to the original name with replaced suffix')
    parser.add_argument('--modules', help='modules to load: local files, remote URLs or mead-ml/hub refs', default=[], nargs='+', required=False)
    parser.add_argument('--no_output_layer', action='store_true', help='If set, we wont store the final layers')
    args = parser.parse_args()

    for module in args.modules:
        import_user_module(module)

    if args.npz is None:
        args.npz = args.model.replace('.pyt', '') + '.npz'

    bl_model = torch.load(args.model, map_location=args.device)
    tpt_embed_dict = bl_model.embeddings

    keys = list(tpt_embed_dict.keys())
    if len(keys) > 1:
        raise Exception(
            "Unsupported model! Multiple embeddings are applied in this model, "
            "but this converter only supports a single embedding"
        )

    tpt_embed = tpt_embed_dict[keys[0]]

    if args.no_output_layer:
        save_tlm_npz(tpt_embed, args.npz, verbose=True)
    else:
        # Monkey patch the embedding to contain an output_layer
        tpt_embed.output_layer = bl_model.output_layer
        save_tlm_output_npz(tpt_embed, args.npz, verbose=True)


if __name__ == '__main__':
    main()

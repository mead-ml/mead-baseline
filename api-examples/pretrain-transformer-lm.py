import logging
import time
import os
from argparse import ArgumentParser
import tempfile
import baseline
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from transformer_utils import on_demand_mlm_masking
from eight_mile.utils import str2bool, write_json, Offsets
import baseline.pytorch.embeddings
import baseline.embeddings
from eight_mile.optz import *
from eight_mile.pytorch.layers import checkpoint_for, rm_old_checkpoints, Average, init_distributed
from eight_mile.pytorch.optz import *
from eight_mile.pytorch.serialize import save_tlm_npz
from baseline.pytorch.lm import TransformerLanguageModel, TransformerMaskedLanguageModel
from baseline.utils import DataDownloader
from transformer_utils import TensorWordDatasetReader, TensorCharDatasetReader, load_data_caching
import numpy as np
import codecs
from collections import Counter

logger = logging.getLogger(__file__)

"""Pre-train a Transformer model in PyTorch

NOTE: Deprecated! This script reads the entire dataset into memory, which may be problematic for large datasets.  For
typical use-cases (BPE single-token, MLM loss models), use pretrain-tlm instead, using the `preproc-tlm`
script to generate fixed contexts upfront.  This is more efficient as it allows us to read and collate full
rows of data without having the dataset in-core, and also because the masking is done upfront.

This file uses Baseline to train a Transformer with PyTorch on multiple GPUs.
It is inspired by: https://github.com/huggingface/naacl_transfer_learning_tutorial/blob/master/pretraining_train.py
This pretraining module has multiple configurations that allow it to support

  * 3 types of pre-training tokenization
    - word
    - subword (based on BERT tokenizer)
    - ELMo (Kim et al 2015) char method
  * pretraining on several datasets including PTB, Wikitext 2 (including raw) and Wikitext 103 (including raw).

If you use `tokens=bpe`, it requires fastBPE.
Otherwise, it depends only on six, numpy, pytorch, and baseline.

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


def create_reader(token_type, nctx, chars_per_word, subword_model_file, subword_vocab_file, subword_special_tokens):
    if token_type == "chars":
        logger.info("Using character input")
        reader = TensorCharDatasetReader(nctx, chars_per_word)
    elif token_type == "words":
        logger.info("Using word input")
        reader = TensorWordDatasetReader(nctx)
    else:
        logger.info("Using subword ({}) input".format(token_type))
        reader = TensorWordDatasetReader(nctx, token_type, subword_model_file, subword_vocab_file,
                                         subword_special_tokens)
    return reader


def get_embed_and_vocab_cache(base_path, dataset_key, token_type, embed_type):
    return os.path.join(base_path, 'preproc-{}-{}-{}.cache'.format(dataset_key, token_type, embed_type))


def load_embed_and_vocab(token_type, reader, dataset, dataset_key, embed_type, d_model, caching):
    base_path = os.path.dirname(dataset['train_file'])
    preproc_cache = get_embed_and_vocab_cache(base_path, dataset_key, token_type, embed_type)
    if caching and os.path.exists(preproc_cache):
        logger.info("Loading cached preprocessing info [%s]", preproc_cache)
        preproc_data = torch.load(preproc_cache)
        vectorizers_mxlen = preproc_data['vectorizers_mxlen']
        if token_type == 'chars':
            char_vectorizer = reader.vectorizers['x']
            tok_vectorizer = reader.vectorizers['y']
            char_vectorizer.max_seen_tok, char_vectorizer.max_seen_char = vectorizers_mxlen['x']
            tok_vectorizer.max_seen = vectorizers_mxlen['y']
        else:
            reader.vectorizers['x'].max_seen = vectorizers_mxlen['x']
    else:
        vocab_sources = [dataset['train_file'], dataset['valid_file']]
        vocabs = reader.build_vocab(vocab_sources)
        valid_num_words = reader.num_words[dataset['valid_file']]
        vectorizers_mxlen = {}
        if token_type == 'chars':
            vectorizers_mxlen['x'] = (reader.vectorizers['x'].max_seen_tok, reader.vectorizers['x'].max_seen_char)
            vectorizers_mxlen['y'] = reader.vectorizers['y'].max_seen
        else:
            vectorizers_mxlen['x'] = reader.vectorizers['x'].max_seen

        logger.info("Read vocabulary")
        embeddings = {}

        # If we are not using chars, then use 'x' for both input and output
        tgt_key = 'x'
        if token_type == 'chars':
            # Write JSON file here and skip this step the second time
            X_CHAR_EMBEDDINGS['embed_type'] = embed_type
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
                                                              embed_type=embed_type)
            logger.info("Using embedding type [%s]", embed_type)
            vocabs['x'] = x_embedding['vocab']
            embeddings['x'] = x_embedding['embeddings']

        preproc_data = {'vocabs': vocabs, 'embeddings': embeddings, 'valid_num_words': valid_num_words,
                        'tgt_key': tgt_key, 'vectorizers_mxlen': vectorizers_mxlen}
        logger.info("Saving preprocessing info [%s]", preproc_cache)
        torch.save(preproc_data, preproc_cache)
    return preproc_data


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--dataset_key", type=str, default='wikitext-2', help="key from DATASETS global")
    parser.add_argument("--train_file", type=str, help='Optional file path to use for train file')
    parser.add_argument("--valid_file", type=str, help='Optional file path to use for valid file')
    parser.add_argument("--dataset_cache", type=str, default=os.path.expanduser('~/.bl-data'),
                        help="Path or url of the dataset cache")
    parser.add_argument("--cache_features", type=str2bool, default=True)
    parser.add_argument("--embed_type", type=str, default='positional',
                        help="register label of the embeddings, so far support positional or learned-positional")
    parser.add_argument("--d_model", type=int, default=410, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2100, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=10, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=16, help="Number of layers")
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")
    parser.add_argument("--tokens", choices=["words", "chars", "bpe", "wordpiece"], default="wordpiece",
                        help="What tokens to use")
    parser.add_argument("--subword_model_file", type=str, help="If using subwords, pass this", default='bert-base-uncased')
    parser.add_argument("--subword_vocab_file", type=str, help="If using subwords with separate vocab file, pass here")
    parser.add_argument("--subword_special_tokens", type=str, nargs='*',
                        help="When using wordpiece vectorizer, this list provide special tokens to the never_split "
                             "argument of BertTokenizer. These special tokens should also be in the customized vocab "
                             "file so that they have their indices.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr_scheduler", type=str, choices=['cosine', 'exponential', 'invtime'],
                        help="choose the type of learning rate decay")
    parser.add_argument("--lr_decay_steps", type=int, default=50000, help="decay steps of lr scheduler")
    parser.add_argument("--lr_decay_rate", type=float, default=0.5, help="decay rate of lr scheduler")
    parser.add_argument("--optim", default="adam", type=str, help="Optimizer to use (defaults to adam)")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--restart_tt", type=str, help="Optional param for legacy checkpoints (step|epoch)")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Num warmup steps")
    parser.add_argument("--mlm", type=str2bool, default=False, help="Use Masked Language Model (MLM) objective")
    parser.add_argument('--rpr_k', help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[0], nargs='+')
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

    if args.tokens == "chars" and args.mlm:
        logger.error("Character composition cannot currently be used with the MLM objective")

    if args.basedir is None:
        args.basedir = 'transformer-{}-{}-{}'.format(args.dataset_key, args.tokens, os.getpid())
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("Cache directory [%s]", args.dataset_cache)

    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")

    if args.distributed:
        args.device = init_distributed(args.local_rank)

    if args.train_file:
        dataset = {'train_file': args.train_file, 'valid_file': args.valid_file}
    else:
        dataset = DataDownloader(DATASETS[args.dataset_key], args.dataset_cache).download()
    if args.subword_special_tokens is None:
        special_tokens = ()
    else:
        special_tokens = tuple(args.subword_special_tokens)
    reader = create_reader(args.tokens, args.nctx, args.chars_per_word, args.subword_model_file,
                           args.subword_vocab_file, special_tokens)

    preproc_data = load_embed_and_vocab(args.tokens, reader, dataset, args.dataset_key,
                                        args.embed_type, args.d_model, args.cache_features)

    vocabs = preproc_data['vocabs']
    if args.mlm:
        mask_from = vocabs['x']
        vocab_size = len(mask_from)
        mask_value = mask_from.get("[MASK]", mask_from.get("<MASK>", -1))
        if mask_value == -1:
            logger.error("We could not find a suitable masking token in the vocab")
            return
    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs['x'], os.path.join(args.basedir, 'vocabs.json'))
    embeddings = preproc_data['embeddings']
    valid_num_words = preproc_data['valid_num_words']
    tgt_key = preproc_data['tgt_key']
    logger.info("Loaded embeddings")

    train_set = load_data_caching(args.tokens, reader, dataset, 'train_file', vocabs, args.cache_features, logger)
    valid_set = load_data_caching(args.tokens, reader, dataset, 'valid_file', vocabs, args.cache_features, logger)
    logger.info("valid. tokens [%s], valid. words [%s]", valid_set.tensors[-1].numel(), valid_num_words)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=args.batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    logger.info("Loaded datasets")

    if len(args.rpr_k) == 0 or args.rpr_k[0] < 1:
        rpr_k = None
    elif len(args.rpr_k) == 1:
        rpr_k = args.rpr_k[0]
    else:
        rpr_k = args.rpr_k

    TLM = TransformerMaskedLanguageModel if args.mlm else TransformerLanguageModel
    model = TLM.create(embeddings,
                       hsz=args.d_model,
                       d_ff=args.d_ff,
                       tie_weights=(args.tokens != 'chars'),
                       dropout=args.dropout,
                       gpu=False,
                       num_heads=args.num_heads,
                       layers=args.num_layers,
                       rpr_k=rpr_k,
                       d_k=args.d_k,
                       src_keys=['x'], tgt_key=tgt_key)
    model.to(args.device)
    loss_function = model.create_loss()
    loss_function.to(args.device)

    logger.info("Loaded model and loss")

    # in this case (train_loader is not iterator) the division by number of gpus is automatically taken care of by
    # torch.DataLoader
    steps_per_epoch = len(train_loader)
    update_on = steps_per_epoch // 10
    logger.info(f"Steps per epoch per GPU: {steps_per_epoch}. Reporting loss every {update_on} steps.")
    if args.lr_scheduler == 'cosine':
        logger.info("Using cosine decay learning rate.")
        lr_decay = CosineDecaySchedulerPyTorch(steps_per_epoch * args.epochs, lr=args.lr)
    else:
        logger.info(f"Using {args.lr_scheduler} decay learning rate with decay steps {args.lr_decay_steps} and decay "
                    f"rate {args.lr_decay_rate}.")
        lr_decay = create_lr_scheduler(lr_scheduler_type=args.lr_scheduler, decay_steps=args.lr_decay_steps,
                                       decay_rate=args.lr_decay_rate, lr=args.lr)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, lr_decay, lr=args.lr)

    global_step = 0
    start_epoch = 0
    if args.restart_from:
        model.load_state_dict(torch.load(args.restart_from))
        vec = args.restart_from.split("-")

        if args.restart_tt:
            tick_type = args.restart_tt
        else:
            tick_type = vec[-2]
        step_num = int(vec[-1].split(".")[0])
        if tick_type == 'epoch':
            start_epoch = step_num
            global_step = start_epoch * steps_per_epoch

        else:
            start_epoch = step_num // steps_per_epoch
            global_step = step_num

        logger.info("Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d, epoch=%d",
                    args.restart_from, global_step, start_epoch + 1)

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

    # This is the training loop
    for epoch in range(start_epoch, args.epochs):
        avg_loss = Average('average_train_loss')
        metrics = {}
        optimizer.zero_grad()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        start = time.time()
        model.train()
        for i, batch in enumerate(train_loader):
            x, y = batch
            inputs = x.to(args.device)
            labels = y.to(args.device)
            if args.mlm:
                inputs, labels = on_demand_mlm_masking(inputs, labels, mask_value, vocab_size)
            inputs = {'x': inputs}
            labels = labels.transpose(0, 1).contiguous()
            logits = model(inputs, None)[0].transpose(0, 1).contiguous()
            if args.mlm:
                loss = loss_function(logits, labels)
            else:
                shift_logits = logits[:-1]
                shift_labels = labels[1:]
                loss = loss_function(shift_logits, shift_labels)
            loss.backward()
            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % update_on == 0:
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
        start = time.time()
        model.eval()
        for batch in valid_loader:
            with torch.no_grad():
                x, y = batch
                inputs = x.to(args.device)
                labels = y.to(args.device)
                if args.mlm:
                   inputs, labels = on_demand_mlm_masking(inputs, labels, mask_value, vocab_size)
                inputs = {'x': inputs}
                labels = labels.transpose(0, 1).contiguous()
                logits = model(inputs, None)[0].transpose(0, 1).contiguous()
                if args.mlm:
                    loss = loss_function(logits, labels)
                else:
                    shift_logits = logits[:-1]
                    shift_labels = labels[1:]
                    loss = loss_function(shift_logits, shift_labels)
                avg_valid_loss.update(loss.item())

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
        logger.info(metrics)

        if args.local_rank < 1:

            # Should probably do this more often
            checkpoint_name = checkpoint_for(model_base, epoch)
            logger.info("Creating checkpoint: %s", checkpoint_name)
            if args.distributed:
                torch.save(model.module.state_dict(), checkpoint_name+'.pth')
                save_tlm_npz(model.module, checkpoint_name+'.npz')
            else:
                torch.save(model.state_dict(), checkpoint_name+'.pth')
                save_tlm_npz(model, checkpoint_name+'.npz')

            rm_old_checkpoints(model_base, epoch)


if __name__ == "__main__":
    train()


import logging
import time
import os
from argparse import ArgumentParser
import tempfile
import baseline
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.pytorch.layers import *
import baseline.pytorch.embeddings
import baseline.embeddings
from paired_utils import *
from eight_mile.optz import *
from eight_mile.pytorch.optz import *
from baseline.utils import DataDownloader


logger = logging.getLogger(__file__)


"""Pre-train a Transformer dual encoder model in PyTorch to predict the second half of the batch

This file uses Baseline to train a Transformer-based ConveRT, but with a variation where instead of predicting
the next turn, it predicts the second half of the batch.  This is more efficient to train than the paired version,
since it doesnt require training with zero-padding, and it also allows the model to scale beyond paired conversations.

Code requirements
  * six
  * numpy
  * pytorch
  * mead-baseline.

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



def create_model(embeddings, d_model, d_ff, dropout, num_heads, num_layers):

    model = PairedModel(embeddings, d_model, d_ff, dropout, num_heads, num_layers)
    logger.info(model)
    return model



def load_data(token_type, reader, dataset, file_key, vocabs, caching):
    cached_file = '{}-{}-dual.cache'.format(dataset[file_key], token_type)
    if caching and os.path.exists(cached_file):
        logger.info("Reloading %s from cached file [%s]", file_key, cached_file)
        loaded = torch.load(cached_file)
    else:
        loaded = reader.load(dataset[file_key], vocabs)
        logger.info("Caching %s to [%s]", file_key, cached_file)
        torch.save(loaded, cached_file)
    return loaded


def create_reader(token_type, nctx, subword_model_file, subword_vocab_file):
    if token_type == "chars":
        raise NotImplementedError("We do not currently support char tokens")
    if token_type == "words":
        logger.info("Using word input")
        reader = SingleSourceTensorWordDatasetReader(nctx)
    else:
        logger.info("Using subword ({}) input".format(token_type))
        reader = SingleSourceTensorWordDatasetReader(nctx, True, subword_model_file, subword_vocab_file)
    return reader


def get_embed_and_vocab_cache(base_path, dataset_key, token_type):
    return os.path.join(base_path, 'preproc-dual-{}-{}.cache'.format(dataset_key, token_type))


def load_embed_and_vocab(token_type, reader, dataset, dataset_key, embed_type, d_model, caching):
    base_path = os.path.dirname(dataset['train_file'])
    preproc_cache = get_embed_and_vocab_cache(base_path, dataset_key, token_type)
    if caching and os.path.exists(preproc_cache):
        logger.info("Loading cached preprocessing info [%s]", preproc_cache)
        preproc_data = torch.load(preproc_cache)
        vectorizers_mxlen = preproc_data['vectorizers_mxlen']
        for k, vectorizer in reader.vectorizers.items():
            vectorizer.max_seen = vectorizers_mxlen[k]
    else:
        vocab_sources = [dataset['train_file'], dataset['valid_file']]
        vocabs = reader.build_vocab(vocab_sources)
        valid_num_words = reader.num_words[dataset['valid_file']]
        vectorizers_maxlen = {}
        for k, vectorizer in reader.vectorizers.items():
            vectorizers_maxlen[k] = vectorizer.max_seen
        logger.info("Read vocabulary")
        embeddings = {}

        # If we are not using chars, then use 'x' for both input and output
        if token_type == 'chars':
            raise NotImplementedError("We do not currently support char tokens")
        else:
            x_embedding = baseline.embeddings.load_embeddings('x',
                                                              dsz=d_model,
                                                              known_vocab=vocabs['x'],
                                                              embed_type=embed_type)
            logger.info("Using embedding type [%s]", embed_type)
            vocabs['x'] = x_embedding['vocab']
            embeddings['x'] = x_embedding['embeddings']

        preproc_data = {'vocabs': vocabs, 'embeddings': embeddings, 'valid_num_words': valid_num_words,
                        'vectorizers_mxlen': vectorizers_maxlen}
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
    parser.add_argument("--num_heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--nctx", type=int, default=256, help="Max input length (half the batch width)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--tokens", choices=["words", "bpe"], default="bpe", help="What tokens to use")
    parser.add_argument("--subword_model_file", type=str, help="If using subwords, pass this")
    parser.add_argument("--subword_vocab_file", type=str, help="If using subwords with separate vocab file, pass here")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--loss", type=str, default='all', choices=['triplet', 'all'])
    parser.add_argument("--update_steps", type=int, default=100,
                        help="The number of steps to take before output a log message")
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

    args = parser.parse_args()

    if args.train_file and not args.valid_file:
        logger.error("If you provide a train_file, you must provide a valid_file")
        return

    if not args.train_file and args.valid_file:
        logger.error("If you provide a valid_file, you must also provide a train_file")
        return

    if args.basedir is None:
        args.basedir = 'dual-transformer-{}-{}-{}'.format(args.dataset_key, args.tokens, os.getpid())
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
        # In an env like k8s with kubeflow each worker will only see a single gpu
        # with an id of 0. If the gpu count is 1 then we are probably in an env like
        # that so we should just use the first (and only) gpu avaiable
        if torch.cuda.device_count() == 1:
            torch.cuda.set_device(0)
            args.device = torch.device("cuda", 0)
        # This program assumes multiprocess/multi-device on a single node. Each
        # process gets a rank (via cli or ENV variable) and uses that rank to select
        # which gpu to use. This only makes sense on a single node, if you had 4
        # processes on 2 nodes where each node has 2 GPUs then the ranks would be
        # 0, 1, 2, 3 but the gpus numbers would be node 0: 0, 1 and node 1: 0, 1
        # and this assignment to gpu 3 would fail. On a single node with 4 processes
        # and 4 gpus the rank and gpu ids will align and this will work
        else:
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.train_file:
        dataset = {'train_file': args.train_file, 'valid_file': args.valid_file}
    else:
        dataset = DataDownloader(DATASETS[args.dataset_key], args.dataset_cache).download()

    reader = create_reader(args.tokens, args.nctx, args.subword_model_file,
                           args.subword_vocab_file)

    preproc_data = load_embed_and_vocab(args.tokens, reader, dataset, args.dataset_key,
                                        args.embed_type, args.d_model, args.cache_features)

    vocabs = preproc_data['vocabs']
    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs['x'], os.path.join(args.basedir, 'vocabs.json'))
    embeddings = preproc_data['embeddings']
    valid_num_words = preproc_data['valid_num_words']
    logger.info("Loaded embeddings")

    train_set = load_data(args.tokens, reader, dataset, 'train_file', vocabs, args.cache_features)
    valid_set = load_data(args.tokens, reader, dataset, 'valid_file', vocabs, args.cache_features)
    logger.info("valid. tokens [%s], valid. words [%s]", valid_set.tensors[-1].numel(), valid_num_words)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=args.batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    logger.info("Loaded datasets")

    model = create_model(embeddings['x'], d_model=args.d_model, d_ff=args.d_ff, dropout=args.dropout,
                         num_heads=args.num_heads, num_layers=args.num_layers)
    model.to(args.device)
    loss_function = model.create_loss(args.loss)
    loss_function.to(args.device)

    logger.info("Loaded model and loss")

    steps_per_epoch = len(train_loader)
    update_on = steps_per_epoch // 10
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
    steps = 0
    model_base = os.path.join(args.basedir, 'dual-checkpoint')

    for epoch in range(start_epoch, args.epochs):
        avg_loss = Average('average_train_loss')
        metrics = {}
        optimizer.zero_grad()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        start = time.time()
        model.train()
        for i, batch in enumerate(train_loader):
            steps += 1
            x, y = batch
            inputs = x.to(args.device)
            labels = y.to(args.device)
            loss = loss_function(inputs, labels)
            loss.backward()
            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % update_on == 0:
                logging.info(avg_loss)

        # How much time elapsed in minutes
        elapsed = (time.time() - start)/60
        train_avg_loss = avg_loss.avg
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_loss'] = train_avg_loss
        avg_valid_loss = Average('average_valid_loss')
        start = time.time()
        model.eval()
        for batch in valid_loader:
            with torch.no_grad():
                x, y = batch
                inputs = x.to(args.device)
                labels = y.to(args.device)
                loss = loss_function(inputs, labels)
                avg_valid_loss.update(loss.item())

        valid_avg_loss = avg_valid_loss.avg

        elapsed = (time.time() - start)/60
        metrics['valid_elapsed_min'] = elapsed
        metrics['average_valid_loss'] = valid_avg_loss
        logger.info(metrics)
        if args.local_rank < 1:
            # Should probably do this more often
            checkpoint_name = checkpoint_for(model_base, epoch)
            logger.info("Creating checkpoint: %s", checkpoint_name)
            if args.distributed:
                torch.save(model.module.state_dict(), checkpoint_name+'.pth')
            else:
                torch.save(model.state_dict(), checkpoint_name+'.pth')

            rm_old_checkpoints(model_base, epoch+1)


if __name__ == "__main__":
    train()


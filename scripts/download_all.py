import os
import argparse
from baseline.utils import read_json
from mead.utils import index_by_label, convert_path
from baseline.utils import EmbeddingDownloader, DataDownloader


parser = argparse.ArgumentParser(description="Download all data and embeddings.")
parser.add_argument("--cache", default="~/.bl-data", type=os.path.expanduser, help="Location of the data cache")
parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
parser.add_argument('--embeddings', help='json library of embeddings', default='config/embeddings.json', type=convert_path)
args = parser.parse_args()


datasets = read_json(args.datasets)
datasets = index_by_label(datasets)

for name, d in datasets.items():
    print(name)
    try:
        DataDownloader(d, args.cache).download()
    except Exception as e:
        print(e)


emb = read_json(args.embeddings)
emb = index_by_label(emb)

for name, e in emb.items():
    print(name)
    try:
        EmbeddingDownloader(e['file'], e['dsz'], e.get('sha1'), args.cache).download()
    except Exception as e:
        print(e)

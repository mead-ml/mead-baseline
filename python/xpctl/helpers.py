import json

__all__ = ["log2json", "read_config"]

def log2json(log):
    s = []
    with open(log) as f:
        for line in f:
            x = line.replace("'", '"')
            s.append(json.loads(x))
    return s


def read_config(config):
    with open(config) as f:
        return json.load(f)

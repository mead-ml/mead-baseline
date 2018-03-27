import argparse
from mead.tasks import Task

parser = argparse.ArgumentParser(description='Train a text classifier')
parser.add_argument('--config', help='JSON Configuration for an experiment', required=True)
parser.add_argument('--datasets', help='json library of dataset labels', default='./config/datasets.json')
parser.add_argument('--embeddings', help='json library of embeddings', default='./config/embeddings.json')
parser.add_argument('--logging', help='json file for logging', default='./config/logging.json')
parser.add_argument('--task', help='task to run', default='classify')
args = parser.parse_args()

task = Task.get_task_specific(args.task, args.logging)
task.read_config(args.config, args.datasets)
task.initialize(args.embeddings)
task.train()



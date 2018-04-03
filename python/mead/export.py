import argparse
import mead

parser = argparse.ArgumentParser(description='Train a text classifier')
parser.add_argument('--config', help='JSON Configuration for an experiment', required=True)
parser.add_argument('--datasets', help='json library of dataset labels', default='./config/datasets.json')
parser.add_argument('--embeddings', help='json library of embeddings', default='./config/embeddings.json')
parser.add_argument('--logging', help='json file for logging', default='./config/logging.json')
parser.add_argument('--task', help='task to run', default='classify')
parser.add_argument('--model', help='model name', required=True)
parser.add_argument('--model_version', help='model_version', default=1)
parser.add_argument('--output_dir', help='output dir', default='./models')

args = parser.parse_args()

task = mead.Task.get_task_specific(args.task, args.logging)
task.read_config(args.config, args.datasets)

exporter = task.create_exporter()
exporter.run(args.model, args.embeddings, args.output_dir, args.model_version)

# https://stackoverflow.com/questions/56659949/saving-a-tf2-keras-model-with-custom-signature-defs

import argparse
import tensorflow as tf
from baseline.tf.embeddings import *
from baseline.model import load_model_for
from baseline.tf.tagger import *
from baseline.tf.classify import *
class ExportModule(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32), tf.TensorSpec(shape=[None,], dtype=tf.int32)])
    def score(self, x, lengths):
        result = self.model({'word': x, 'lengths': lengths})
        return { "scores": result }


def main():
    parser = argparse.ArgumentParser(description='Export TF2 model')
    parser.add_argument('--task', help='Task name', required=True, type=str)
    parser.add_argument('--model_file', help='A model file', required=True, type=str)
    parser.add_argument('--export_path', required=True)
    args = parser.parse_args()

    SET_TRAIN_FLAG(False)
    model = load_model_for(args.task, args.model_file)
    export_module = ExportModule(model)
    tf.saved_model.save(export_module, args.export_path, signatures={ "score": export_module.score })



if __name__ == '__main__':
    main()

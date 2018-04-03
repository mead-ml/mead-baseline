from mead.tasks import *
from mead.utils import *


class Exporter(object):

    def __init__(self, task):
        super(Exporter, self).__init__()
        self.task = task

    def run(self, model_file, embeddings, output_dir, model_version, **kwargs):
        pass

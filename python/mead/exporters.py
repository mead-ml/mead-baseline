from mead.tasks import *
from mead.utils import *
from baseline.utils import export

__all__ = []
exporter = export(__all__)

@exporter
class Exporter(object):

    def __init__(self, task):
        super(Exporter, self).__init__()
        self.task = task

    def run(self, model_file, embeddings, output_dir, model_version, **kwargs):
        pass

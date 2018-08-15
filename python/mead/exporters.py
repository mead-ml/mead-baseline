from mead.tasks import *
from mead.utils import *
from baseline.utils import export, import_user_module

__all__ = []
exporter = export(__all__)

@exporter
class Exporter(object):

    def __init__(self, task):
        super(Exporter, self).__init__()
        self.task = task

    def run(self, model_file, embeddings, output_dir, model_version, **kwargs):
        pass


def create_exporter(task, exporter_type):
    if exporter_type == 'default':
        return task.ExporterType
    else:
        mod = import_user_module("exporter", exporter_type)
        return mod.create_exporter(task, exporter_type)

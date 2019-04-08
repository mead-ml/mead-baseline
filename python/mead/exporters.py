from baseline.utils import export, optional_params

__all__ = []
exporter = export(__all__)


@exporter
class Exporter(object):

    def __init__(self, task, **kwargs):
        super(Exporter, self).__init__()
        self.task = task

    @classmethod
    def preproc_type(cls):
        return 'client'

    def run(self, model_file, output_dir, project=None, name=None, model_version=None, **kwargs):
        pass


BASELINE_EXPORTERS = {}


@exporter
@optional_params
def register_exporter(cls, task, name=None):
    """Register an exporter

    Use this pattern if you want to provide an override to a `Exporter` class.

    """
    if name is None:
        name = cls.__name__

    if task not in BASELINE_EXPORTERS:
        BASELINE_EXPORTERS[task] = {}

    if name in BASELINE_EXPORTERS[task]:
        raise Exception('Error: attempt to re-defined previously registered handler {} in exporter registry'.format(name))

    BASELINE_EXPORTERS[task][name] = cls
    return cls


def create_exporter(task, name=None, **kwargs):
    return BASELINE_EXPORTERS[task.task_name()][name](task, **kwargs)

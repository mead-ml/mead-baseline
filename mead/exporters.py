import os
from baseline.utils import exporter, optional_params

__all__ = []
export = exporter(__all__)


@export
class Exporter(object):

    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task

    @classmethod
    def preproc_type(cls):
        return 'client'

    def _run(self, model_file, output_dir, project=None, name=None, model_version=None, **kwargs):
        raise NotImplementedError

    def run(self, model_file, output_dir, project=None, name=None, model_version=None, **kwargs):
        client_loc, server_loc = self._run(
            model_file,
            output_dir,
            project=project,
            name=name,
            model_version=model_version,
            **kwargs
        )
        if model_version is None:
            try:
                model_version = int(os.path.basename(client_loc))
            except ValueError:
                pass
        msg = {
            "client_bundle": client_loc,
            "server_bundle": server_loc,
            "project": project,
            "name": name,
            "version": model_version
        }
        for rep in self.task.reporting:
            rep.step(msg, 0, 'Export', 'EXPORT')
        self.task._close_reporting_hooks()
        return client_loc, server_loc


BASELINE_EXPORTERS = {}


@export
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

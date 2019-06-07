try:
    from baseline.mime_type import *
    from baseline.utils import *
    from baseline.vectorizers import *
    from baseline.w2v import *
    from baseline.confusion import *
    from baseline.data import *
    from baseline.reader import *
    from baseline.progress import *
    from baseline.reporting import *
    from baseline.model import *
    from baseline.embeddings import *
    from baseline.services import *
    from baseline.train import *

    logger = get_console_logger('baseline', env_key='BASELINE_LOG_LEVEL')
    report_logger = get_console_logger('baseline.reporting', env_key='BASELINE_LOG_LEVEL')
except ImportError:
    pass
from baseline.version import __version__

from ce_standards.version import __version__

__version__ = __version__

from cengine.client import Client
from cengine.models import Workspace
from cengine.models import Datasource
from cengine.pipeline_config import PipelineConfig, Split, Categorize, Index, \
    Features, MethodLists, Method, Trainer, Preprocessing, Timeseries

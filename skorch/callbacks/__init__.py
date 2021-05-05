"""This module serves to elevate callbacks in submodules to the
skorch.callback namespace. Remember to define `__all__` in each
submodule.

"""

# pylint: disable=wildcard-import

from .base import *
from .logging import *
from .regularization import *
from .scoring import *
from .training import *
from .lr_scheduler import *


__all__ = [
    'BatchScoring',
    'Callback',
    'Checkpoint',
    'EarlyStopping',
    'EpochScoring',
    'EpochTimer',
    'Freezer',
    'GradientNormClipping',
    'Initializer',
    'LRScheduler',
    'LoadInitState',
    'NeptuneLogger',
    'ParamMapper',
    'PassthroughScoring',
    'PrintLog',
    'ProgressBar',
    'TrainEndCheckpoint',
    'TensorBoard',
    'SacredLogger',
    'Unfreezer',
    'WandbLogger',
    'WarmRestartLR',
    'MLflow',
]

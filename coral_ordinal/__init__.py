from .version import __version__

from .layer import CoralOrdinal
from .loss import OrdinalCrossEntropy
from .metrics import MeanAbsoluteErrorLabels
from .activations import ordinal_softmax

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
    "CoralOrdinal",
    "MeanAbsoluteErrorLabels",
    "OrdinalCrossEntropy",
    "ordinal_softmax",
]

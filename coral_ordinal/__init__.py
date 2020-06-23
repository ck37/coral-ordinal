from .version import __version__

from .layer import CoralOrdinal
from .loss import OrdinalCrossEntropy
from .utils import logits_to_probs
from .metrics import MeanAbsoluteErrorLabels

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
  'CoralOrdinal',
  'MeanAbsoluteErrorLabels',
  'OrdinalCrossEntropy',
  'logits_to_probs',
]

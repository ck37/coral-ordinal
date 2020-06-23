from .version import __version__

from .layer import CoralOrdinal
from .loss import OrdinalCrossEntropy
from .utils import logits_to_probs

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
  'CoralOrdinal',
  'OrdinalCrossEntropy',
  'logits_to_probs'
]

from .version import __version__

from .layer import CoralOrdinal
from .loss import CoralOrdinalLoss
from .loss import ordinal_loss
from .utils import logits_to_probs

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
  'CoralOrdinal',
  'CoralOrdinalLoss',
  'ordinal_loss',
  'logits_to_probs'
]

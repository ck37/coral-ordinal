from .version import __version__

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
  'CoralOrdinal',
  'CoralOrdinalLoss',
  'ordinal_loss',
  'logits_to_probs'
]

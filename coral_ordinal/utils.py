import pandas as pd
from scipy import special

# Preds come out of model.predict()
def logits_to_probs(preds, num_classes):
  preds = pd.DataFrame(preds)
 
  # Create a new dataframe to store the probabilities.
  probs = pd.DataFrame(0., index = range(preds.shape[0]), columns = range(num_classes))

  # First, get probability predictions out of the cumulative logits.
  # Column 0 is Probability that y > 0, so Pr(y = 0) = 1 - Pr(y > 0)
  # Pr(Y = 0) = 1 - s(logit for column 0)
  probs[0] = 1. - special.expit(preds[0])

  # Pr(y = 9) = Pr(y > 8)
  probs[num_classes - 1] = special.expit(preds[num_classes - 2])

  # For the other columns, the probability is:
  # Pr(y = k) = Pr(y > k) - Pr(y > k - 1)
  if num_classes > 2:
    for val in range(1, num_classes - 1):
      probs[val] = special.expit(preds[val - 1]) - special.expit(preds[val])
  
  return probs

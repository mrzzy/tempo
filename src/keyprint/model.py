#
# tempo
# keyprint model
#

import numpy as np
import tensorflow as tf

## metric
# defines the distance  metric used to compare embed_target with embed_ref
def distance(embed_ref, embed_target):
    return tf.norm(embed_ref - embed_target, 2)

# Computes the constructive loss of the pred_distance computed between embeddings
# If label is 1 - computes a higher loss if pred_distance is higher
# If label is 0
# - computes a loss if pred_distance is less then margin
# - the lower the pred_distance, the higher the loss
# Returns the computed loss
def contrastive_loss(label, pred_distance, margin=1):
    pos_loss = pred_distance
    neg_loss = max(0, margin - pred_distance)

    return label * pos_loss + (1 - label) * neg_loss

## model builing utilities

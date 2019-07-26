#
# tempo
# keyprint model
#

import numpy as np
import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Conv1D, Activation, BatchNormalization,
                                     AlphaDropout, GlobalAvgPool1D, LeakyReLU,
                                     MaxPool1D, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler

from dataprep import FEATURE_VEC_LEN

N_FEATURES = 5

## metric
# defines the distance  metric used to compare embedding with embed_ref
def distance(embed_ref, embed):
    return tf.norm(embed_ref - embed, 2)

# Computes the constructive loss of the pred_distance computed between embeddings
# If label is 1 - computes a higher loss if pred_distance is higher
# If label is 0
# - computes a loss if pred_distance is less then margin
# - the lower the pred_distance, the higher the loss
# Returns the computed loss
def contrastive_loss(label, pred_distance, margin=1):
    pos_loss = pred_distance
    neg_loss = tf.maximum(0.0, margin - pred_distance)

    return tf.reduce_sum(label * pos_loss + (1 - label) * neg_loss)


# Accuracy metric
# Returns true if predicts the label correctly using threshold as a decision 
# boundary, otherwise false
def accuracy(label, pred_distance, threshold=0.5):
    prediction = (pred_distance < threshold)
    return tf.reduce_mean(tf.equal(label, prediction))

## model builing utilities
# Build a convolution block with the given parameters
# n_filters - no. of convolution filters
# filter_size - size of the convolution filter
# n_layers - no. of convolution layers
# activation - activaation function
# batch_norm - if true uses batch normalisation
# l2_lambda - l2 regularization
# dropout_prob - dropout probability
# return block that takes in input tensor x
def conv_block(n_filters, filter_size, n_layers, activation, batch_norm=None,
               l2_lambda=0, dropout_prob=0):
    def block(x):
        for i in range(n_layers):
            x = Conv1D(n_filters, filter_size,
                       padding="same",
                       kernel_regularizer=l2(l2_lambda))(x)
            if batch_norm: x = BatchNormalization()(x)
            x = activation()(x)
            if dropout_prob: AlphaDropout(dropout_prob)
        return x
    return block

# Build an dense block with the given parameters
# n_units - no. of units in each dense layer
# n_layers - no. of convolution layers
# activation - activaation function
# batch_norm - if true uses batch normalisation
# l2_lambda - l2 regularization
# dropout_prob - dropout probability
# return block that takes in input tensor x
def dense_block(n_units, n_layers, activation, batch_norm=None,
               l2_lambda=0, dropout_prob=0):
    def block(x):
        for i in range(n_layers):
            x = Dense(n_units,
                      kernel_regularizer=l2(l2_lambda))(x)
            if batch_norm: x = BatchNormalization()(x)
            x = activation()(x)
            if dropout_prob: AlphaDropout(dropout_prob)
        return x
    return block

# Build an encoder with the given parameters
# n_input_dim - dimension of input feature vecto
# n_encoding_dim - ouptut encoding dimension
# n_conv_block - no. of convolution blocks
# n_conv_layers - layers per conv block
# n_conv_filters - filters per conv block
# conv_filter size - size of the filter per conv blokc
# n_dense_layers - no. of dense layers
# n_dense_units - no. of dense units per layer
# activation - activaation function
# batch_norm - if true uses batch normalisation
# l2_lambda - l2 regularization
# dropout_prob - dropout probability
# Returns encoder model
def build_encoder(
          n_input_dim,
          n_encoding_dim,
          n_conv_block,
          n_conv_layers,
          n_conv_filters,
          conv_filter_size,
          n_dense_layers,
          n_dense_units,
          activation,
          batch_norm=False,
          l2_lambda=0,
          dropout_prob=0):
    ##  build model graph
    # convolution part
    input_op = Input([FEATURE_VEC_LEN, n_input_dim])
    x = input_op
    for i in range(n_conv_block):
        x = conv_block(n_conv_filters[i],
                       conv_filter_size[i],
                       n_conv_layers[i],
                       activation,
                       batch_norm,
                       l2_lambda,
                       dropout_prob)(x)
        x = MaxPool1D((2,))(x)
    x = GlobalAvgPool1D()(x)
    # dense part
    x = dense_block(n_dense_units,
                    n_dense_layers,
                    activation,
                    batch_norm,
                    l2_lambda,
                    dropout_prob)(x)
    # produce encoding
    encoding = Dense(n_encoding_dim)(x)
    return Model(input_op, encoding)

def build(n_input_dim, encoder):
    ref_input = Input([None, n_input_dim])
    eval_input = Input([None, n_input_dim])
    # pass input through encoder to predict embeddings
    ref_embed = encoder(ref_input)
    eval_embed = encoder(eval_input)
    # compute distance between predicted embeddings
    pred_distance  = Lambda((lambda X: distance(*X)))([ref_embed, eval_embed])

    return Model([ref_input, eval_input], pred_distance)\

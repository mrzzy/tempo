#
# Tempo
# Greyc Dataset
# Data Prepration Utilties
#

import random

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

from multiprocessing import Pool, cpu_count

FEATURE_VEC_LEN = 64

# available features
KEYSTROKE_FEATURES = [
    "keycodes",
    "relative_press_timestamps",
    "relative_release_timestamps",
    "press_to_press_times",
    "press_to_release_times",
    "release_to_press_times",
    "release_to_release_times"
]

# Extracts keystroke features from the raw_presss an raw_release keystroke  features
# Arguments raw_press an raw_release are lists of [<keycode>,<timestamp>]
# If specified, feature_names specify the names features are extracted.
# Returns a features np array of extracted features of shape 
# [<timestep>, <feature>]
def extract_keystroke_features(raw_press, raw_release,
                               feature_names=KEYSTROKE_FEATURES):
    raw_press = np.asarray(raw_press, dtype="int")
    raw_release = np.asarray(raw_release, dtype="int")

    # unpack the features
    press_timestamps = raw_press[:, 1]
    release_timestamps = raw_release[:, 1]

    # ensure that sequences have the same length
    max_len = max(len(press_timestamps), len(release_timestamps))
    press_timestamps = np.pad(press_timestamps,
                              [[0, max_len - len(press_timestamps)]],
                              "constant")
    release_timestamps = np.pad(release_timestamps,
                              [[0, max_len - len(release_timestamps)]],
                              "constant")

    # compute shifted timestamps
    shifted_press_timestamps = np.roll(press_timestamps, 1)
    shifted_release_timestamps = np.roll(release_timestamps, 1)

    ## extract features
    features = []
    # keycodes
    if "keycodes" in feature_names: features.append(raw_press[:, 0])

    # relative timestamps
    if "relative_press_timestamps" in feature_names:
        features.append(press_timestamps - press_timestamps[0])
    if "relative_release_timestamps" in feature_names:
        features.append(release_timestamps - release_timestamps[0])

    # extract press to press timings
    if "press_to_press_times" in feature_names:
        press_to_press_times = press_timestamps - shifted_press_timestamps
        press_to_press_times[0] = 0
        features.append(press_to_press_times)

    # extract press to release timings
    if "press_to_release_times" in feature_names:
        press_to_release_times = press_timestamps - shifted_release_timestamps
        press_to_release_times[0] = 0
        features.append(press_to_release_times)

    # extract release to press timings
    if "release_to_press_times" in feature_names:
        release_to_press_times = release_timestamps - shifted_press_timestamps
        release_to_press_times[0] = 0

    # extract release to release timings
    if "release_to_release_times" in feature_names:
        release_to_release_times =  release_timestamps - shifted_release_timestamps
        release_to_release_times[0] = 0

    ## combine features into feature matrix
    feature_matrix = np.stack(features).T

    return feature_matrix

## Generating pairwise features
def generate_pair(arg):
    i, keystroke_features, meta_df, real_user_indexes = arg
    userid = meta_df.loc[i, "userid"]
    target_userid = meta_df.loc[i, "target_userid"]
    label = 1 if userid == target_userid else 0

    # randomly choose other other pair
    if not target_userid in real_user_indexes:
        return None

    ref_i = random.choice(real_user_indexes[target_userid])
    return [keystroke_features[ref_i],  keystroke_features[i]], label

def get_real_user_index(arg):
    userid, meta_df = arg
    real_user_index = meta_df.index[
        (meta_df["userid"] == meta_df["target_userid"])
        & (meta_df["userid"] == userid) ]

    return userid, real_user_index

# Generate pairwise features
def generate_pair_features(keystroke_features, meta_df):
    proc = Pool(cpu_count())
    # compute the real indexes for each user
    args = [[i, meta_df] for i in meta_df["userid"]]
    results = proc.map(get_real_user_index, args)
    real_user_indexes = dict(results)

    ## generate features pais
    args = [[i, keystroke_features, meta_df, real_user_indexes] for i in meta_df.index]
    results = proc.map(generate_pair, args)
    results = [ r for r in  results if not r is None ]
    feature_pairs, labels = zip(*results)

    return np.asarray(feature_pairs), np.asarray(labels)

## Transformers
# Transformer for extract_keystroke_features()
class KeystrokeFeatureExtractor(BaseEstimator, TransformerMixin):
    # create a feture extractor that extracts the features with the given names 
    def __init__(self, feature_names=KEYSTROKE_FEATURES):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        extract_fn = (lambda x:
                      extract_keystroke_features(*x, feature_names=self.feature_names))
        features = [ extract_fn(x) for x in X ]

        # Normalise features by padding feature vectors
        features = pad_sequences(features,
                                 maxlen=FEATURE_VEC_LEN,
                                 dtype="float64", padding="post")
        return features

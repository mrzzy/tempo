#
# Tempo
# Data Prepration Utilties
#

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

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

## Transformers
# Transformer for extract_keystroke_features()
class KeystrokeFeatureExtractor(BaseEstimator, TransformerMixin):
    # create a feture extractor that extracts the features with the given names 
    def __init__(self, feature_names=KEYSTROKE_FEATURES):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        extract_fn = (lambda x: extract_keystroke_features(*x, feature_names=self.feature_names))
        return [ extract_fn(x) for x in X ]

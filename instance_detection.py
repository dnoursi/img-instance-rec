

import numpy as np
import matplotlib.pyplot as plt

import imutils

from util import *
from visualize import *
from hw2 import *


# Return 5 rescaled versions of a single image
def many_sizes(img):
    result = []
    height0 = img.shape[0]

    scales = [.5, .75, 1., 1.25, 1.5]
    heights = [int(scale * height0) for scale in scales]

    for height in heights:
        result.append(imutils.resize(img, height = height))

    return result

# Input: a scene image, and many instance images
# Output: indices of those instance images which have been detected in the scene image
def detect_instance(scene_img, instance_imgs):
    # N is number of instance points desired per image
    N_instance = 20
    N_scene = N_instance * 10
    # Maps instance index: instance feature vectors
    instance_feats_map = []
    instance_feats = []
    for i, img0 in enumerate(instance_imgs):
        xs0, ys0, scores0 = find_interest_points(img0, N_instance, 1.0)
        feats0, _ = extract_features(img0, xs0, ys0, 1.0)
        instance_feats += feats0
        assert len(feats0) == N_instance
        instance_feats_map += ([i] * N_instance)

    # So instance_feats_map[j] stores which instance feature vector j came from
    # Note: could be eliminated with "N_instance" based index arithmetic

    # Create all reasonable scene feats
    scene_imgs = many_sizes(scene_img)
    scene_feats = []
    for img0 in scene_imgs:
        xs0, ys0, scores0 = find_interest_points(img0, N, 1.0)
        feats0, _ = extract_features(img0, xs0, ys0, 1.0)
        scene_feats += feats0

    # NN
    # TODO is scores=None okay?
    scores0 = scores1 = None
    # Returns: matches:
    # a numpy array of shape (N0,) containing, for each feature in feats0,
    # the index of the best matching feature in feats1
    matches, _ = match_features(scene_feats, instance_feats, scores0, scores1)
    # detection_scores = [(0,i) for i in range(len(instance_imgs))]
    detection_scores = [0] * len(instance_imgs)
    for match in matches:
        instance_index = instance_feats_map[match]
        # detection_scores[instance_index][0] += 1
        detection_scores[instance_index] += 1

    detections = []
    # Denote an object instance as detected if at least "80%" of the descriptor vectors
    #  for this instance were the top match of some scene descriptor vector at some scale
    # So threshold represents this 80% value.
    # Note that each scene has descriptors created at multiple scales, so
    #  thresh could be above 1.0 and be reached
    score_thresh = .8
    for instance_index, score in enumerate(detection_scores):
        if score > .8 * N_instance:
            detections.append(instance_index)

    return detections

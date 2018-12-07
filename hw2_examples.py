## Homework 2
##
## For this assignment, you will design and implement interest point detection,
## feature descriptor extraction, and feature matching.  From matched
## descriptors in two different views of the same scene, you will predict a
## transformation (here simplified to translation only) relating the two
## views.
##
## As interest points, feature descriptors, and matching criterion all involve
## design choices, there is flexibility in the implementation details, and many
## reasonable solutions.
##
## As was the case for homework 1, your implementation should be restricted to
## using low-level primitives in numpy.
##
## See hw2.py for detailed descriptions of the functions you must implement.
##
## In addition to submitting your implementation in hw2.py, you will also need
## to submit a writeup in the form of a PDF (hw2.pdf) which briefly explains
## the design choices you made and any observations you might have on their
## effectiveness.
##
## Resources:
##
##  - A working implementation of Canny edge detection is available (canny.py)
##    for your use, should you wish to build your feature descriptor based on
##    Canny edges.  Note that the output is changed slightly: this detector
##    returns a nonmax-suppressed edge map, and edge orientations.
##
##    Alternatively, you can use your own implementation of Canny edges from
##    homework 1.  If so, feel free to replace the provided canny.py with
##    your own.
##
##  - Some functions for visualizing interest point detections and feature
##    matches are provided (visualize.py).
##
## Submit:
##
##    hw2.py              - your implementation of homework 2
##    hw2.pdf             - your writeup describing your design choices
##    canny.py (optional) - submit only if you modify or replace the
##                          provided implementation

import numpy as np
import matplotlib.pyplot as plt

from util import *
from visualize import *
from hw2 import *

## Examples.
##
## Each subdirectory of data/ contains a set of images of the same scene from
## different views (with largely translational motion).  By extracting feature
## descriptors and finding correspondences, you will be able to estimate the
## parameters of this translation.
##
## Feel free to experiment with different examples.
##
## Images are from the Adobe Panoramas Data Set.

#img0 = load_image('data/halfdome_small/halfdome-00.png')
#img1 = load_image('data/halfdome_small/halfdome-01.png')

#img0 = load_image('data/goldengate/goldengate-02.png')
#img1 = load_image('data/goldengate/goldengate-03.png')

#img0 = load_image('data/rio/rio-47.png')
#img1 = load_image('data/rio/rio-48.png')

img0 = load_image('data/shanghai/shanghai-23.png')
img1 = load_image('data/shanghai/shanghai-24.png')

## Problem 1 - Interest Point Operator
##             (12 Points Implementation + 3 Points Write-up)
##
## (A) Implement find_interest_points() as described in hw2.py    (12 Points)
## (B) Include (in hw2.pdf) a brief description of the design      (3 Points)
##     choices you made and their effectiveness.

N = 200
xs0, ys0, scores0 = find_interest_points(img0, N, 1.0)
xs1, ys1, scores1 = find_interest_points(img1, N, 1.0)

plot_interest_points(img0, xs0, ys0, scores0)
plot_interest_points(img1, xs1, ys1, scores1)
plt.show(block = False)

## Problem 2 - Feature Descriptor Extraction
##             (12 Points Implementation + 3 Points Write-up)
##
## (A) Implement extract_features() as described in hw2.py        (12 Points)
## (B) Include (in hw2.pdf) a brief description of the design      (3 Points)
##     choices you made and their effectiveness.

feats0 = extract_features(img0, xs0, ys0, 1.0)
feats1 = extract_features(img1, xs1, ys1, 1.0)

## Problem 3 - Feature Matching
##             (7 Points Implementation + 3 Points Write-up)
##
## (A) Implement match_features() as described in hw2.py           (7 Points)
## (B) Include (in hw2.pdf) a brief description of the scheme      (3 Points)
##     you used for scoring the quality of a match.

matches, match_scores = match_features(feats0, feats1, scores0, scores1)

threshold = 1.0 # adjust this for your match scoring system
plot_matches(img0, img1, xs0, ys0, xs1, ys1, matches, match_scores, threshold)
plt.show(block = False)

## Problem 4 - Hough Transform
##             (7 Points Implementation + 3 Points Write-up)
##
## (A) Implement hough_votes() as described in hw2.py              (7 Points)
## (B) Briefly describe (in hw2.pdf) your binning strategy.        (3 Points)

tx, ty, votes = hough_votes(xs0, ys0, xs1, ys1, matches, match_scores)

show_overlay(img0, img1, tx, ty)
#plt.show(block = False)
plt.show()


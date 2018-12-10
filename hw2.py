# David Noursi

import numpy as np
import canny
import imutils

"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 4.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at tinthe default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points

   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)
"""
def find_interest_points(image, max_points = 200, scale = 1.0, img_dir = None):
   # check that image is grayscale
    assert image.ndim == 2, 'image should be grayscale'

    dx, dy = canny.sobel_gradients(image)

    or_x, or_y = dir_img(dx, dy, image)

    if img_dir is not None :
      angle = (or_x - img_dir) * 180.0/np.pi
      image = imutils.rotate(image, angle)


    win_size = int(scale)

    xr = dx.shape[0]
    yr = dx.shape[1]

    score_tuples = []

    for ix in range(win_size, xr-win_size):
        for iy in range(win_size, yr-win_size):
            localx = dx[ix-win_size: ix+win_size+1, iy-win_size: iy+win_size+1]
            localy = dy[ix-win_size: ix+win_size+1, iy-win_size: iy+win_size+1]

            localx = np.ndarray.flatten(localx)
            localy = np.ndarray.flatten(localy)

            xdotx = np.dot(localx, localx)
            ydoty = np.dot(localy, localy)
            xdoty = np.dot(localx, localy)

            #score = ((xdotx * ydoty) - (xdoty ** 2)) / (xdotx + ydoty)
            alpha = 0.05
            score = ((xdotx * ydoty) - (xdoty**2)) - alpha * ( (xdotx + ydoty)**2)
            score_tuples.append((score, ix, iy))

    # Max-sort by the first element of the tuple (score)
    score_tuples.sort(reverse = True)

    # Now obtain top 200 points, with nonmax suppression

    xs = []
    ys = []
    scores = []
    # Sets quickly check presence (if _ in banned)
    banned = set()

    # Nonmax suppression window vs original score window?
    k = 1
    offsets = list(range(-k*win_size, k*win_size + 1))

    for tupl in score_tuples:
        if len(scores) == max_points:
            break

        score, x, y = tupl
        if (x, y) in banned:
            continue

        scores.append(score)
        xs.append(x)
        ys.append(y)

        for ox in offsets:
            for oy in offsets:
                banned.add((x + ox, y + oy))
    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys, scores, ((or_x + or_y)/2) #should be the same exact number, but for good measure (probably unnnecasary)

"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""

# Compute coords of unit offset vector, then fill histograms of joint coord values
def energy_bins(window):
    offset_x = np.round(np.cos(window)).astype(int);
    offset_y = np.round(np.sin(window)).astype(int);
    # Turns [-1,0,1] into valid list indices
    offset_x += 1
    offset_y += 1

    results = np.zeros((3,3))
    for ox,oy in zip(np.ndarray.flatten(offset_x), np.ndarray.flatten(offset_y)):
        results[ox][oy] += 1

    return np.ndarray.flatten(results)
    # Order of vector coordinates does not matter; flatten is deterministic, therefore repeatable

def extract_features(image, xs, ys, scale = 1):
   # check that image is grayscale
    assert image.ndim == 2, 'image should be grayscale'
    assert len(xs) == len(ys), 'xs and ys correspond'

    dx, dy = canny.sobel_gradients(image)
    theta = np.arctan2(dy, dx)

    # Take 8 bins around the center bin, where each bin has volume 4 * b_e_r**2
    bin_edge_radius = int(scale)
    bin_center_offset = int(2* scale) + 1
    offsets = [-1, 0, 1]

    feats = []
    imshape = image.shape
    for x, y in zip(xs, ys):
        # ob is Offset Bin
        feat = []
        for obx in offsets:
            for oby in offsets:
                if obx == 0 == oby:
                    continue
                # bc is Bin Center
                bcx = x + obx*bin_center_offset
                bcy = y + oby*bin_center_offset
                xlow = bcx - bin_edge_radius
                xhigh = bcx + bin_edge_radius + 1
                ylow = bcy - bin_edge_radius
                yhigh = bcy + bin_edge_radius + 1
                if xlow < 0 or ylow < 0 or xhigh > imshape[0] or yhigh > imshape[1]:
                    feat = np.concatenate((feat, np.zeros((9))))
                else:
                    window = theta[xlow: xhigh, ylow: yhigh]
                    histogram = energy_bins(window)
                    feat = np.concatenate((feat,histogram))
        feats.append(feat)

    feats = np.array(feats)
    assert feats.shape == (len(xs), 72), "check feats.shape before returning"

    return feats

    ##########################################################################
    # TODO: YOUR CODE HERE
    #raise NotImplementedError('extract_features')
    ##########################################################################

#compute averge orientation at each pixel 
def dir_img(dx, dy, img):
  for x in range(len(dx)):
    for y in range(len(dx[x])):
      dx[x][y] = np.arctan2(y, x) % np.pi

  orientation_x =  np.average(dx)

  for x in range(len(dy)):
    for y in range(len(dy[x])):
      dy[x][y] = np.arctan2(y, x) % np.pi

  orientation_y =  np.average(dy)

  print(orientation_x, orientation_y)

  return orientation_x, orientation_y

"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   the second set in general contains many more elements than the first set
   the second set is indexed by a kd tree

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match.  It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())

   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""

import collections

# https://en.wikipedia.org/wiki/K-d_tree
TreeNode = collections.namedtuple("TreeNode", ["axis", "index", "left", "right"])


# at a layer we have: [axis, splitt3ing node index, sub tree if lesser than, sub tree if greater than]
def make_kd_tree(points):
    k = len(points[0])
    depth = 0
    values = [(p,i) for i,p in enumerate(points)]
    return kd_tree(values, 0, k)

def kd_tree(values, depth, k):
    if not values:
        return None

    axis = depth % k
    axisValues = [(p[axis],i) for (p,i) in values]
    axisValues.sort()
    median = len(axisValues) // 2 # choose median

    # Create node and construct subtrees
    index = axisValues[median][1]

    leftIndices = [value[1] for value in axisValues[:median]]
    leftValues = [value  for value in values if value[1] in leftIndices]
    leftTree = kd_tree(leftValues, depth + 1, k)
    left = leftTree

    rightIndices = [value[1] for value in axisValues[median+1:]]
    rightValues = [value  for value in values if value[1] in rightIndices]
    rightTree = kd_tree(rightValues, depth + 1, k)
    right = rightTree

    result = TreeNode(axis=axis, index=index, left=left, right=right)
    return result



def traverse(vector, treeRoot, treeVectors):
    binsHeap = []
    currentNode = treeRoot

    soln = currentNode
    solnDist = vecDist(vector, treeVectors, soln.index)

    while currentNode:
        # Check if we've found the best current solution
        distHere = vecDist(vector, treeVectors, currentNode.index)
        if distHere < solnDist:
            soln = currentNode
            solnDist = distHere

        # Traverse tree
        dimDist = vector[currentNode.axis] - treeVectors[currentNode.index][currentNode.axis]
        if dimDist > 0:
            binsHeap.append((abs(dimDist), currentNode.right))
            currentNode = currentNode.left
        else:
            binsHeap.append((abs(dimDist), currentNode.left))
            currentNode = currentNode.right

    binsHeap.sort(key=lambda x:x[0])
    return soln, binsHeap

def nearestNeighbor(vector, treeRoot, treeVectors):
    if not treeRoot:
        return None
    soln, binsHeap = traverse(vector, treeRoot, treeVectors)
    solnDist = vecDist(vector, treeVectors, soln.index)

    # BBF ensure that we've found the best NN
    for (minBinDist, binTree) in binsHeap:
        # All potential bins checked
        if minBinDist > solnDist:
            break
        altSoln = nearestNeighbor(vector, binTree, treeVectors)
        if altSoln:
            altSolnDist = vecDist(vector, treeVectors, altSoln.index)
            if altSolnDist < solnDist:
                soln = altSoln
                solnDist = altSolnDist

    return soln

def vecDist(v1, v2list, v2index):
    v2 = v2list[v2index]
    return np.linalg.norm(v1 - v2)

def match_features(feats0, feats1, scores0, scores1):
    matches = []
    scores = []

    kdTree = make_kd_tree(feats1)
    for feat0 in feats0:
        matched = nearestNeighbor(feat0, kdTree, feats1)
        matches.append(matched.index)
    matches = np.array(matches, np.int)
    return matches, None

def hough_votes(xs0, ys0, xs1, ys1, matches, scores):

    scale = 3
    bins = {}
    for i in range(len(xs0)):
        i1 = matches[i]

        x0 = xs0[i]
        x1 = xs1[i1]
        binx = int(-(x1 - x0)/scale)

        y0 = ys0[i]
        y1 = ys1[i1]
        biny = int(-(y1 - y0)/scale)

        if (binx, biny) not in bins: #bins.keys():
            bins[(binx, biny)] = 0
        bins[(binx, biny)] += 1 # scores[i]

    tx, ty = max(bins, key=bins.get)
    tx *= scale
    ty *= scale
    votes = bins

    ##########################################################################
    # TODO: YOUR CODE HERE
    #raise NotImplementedError('hough_votes')
    ##########################################################################
    return tx, ty, votes

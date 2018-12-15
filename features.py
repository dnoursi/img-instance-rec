# David Noursi

import numpy as np
import canny
import collections

"""
   Interest point detection - utilizes the harris corner detector operator, 
    computing score as well (artificat of earlier implementation)

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
def find_interest_points(image, max_points = 200, scale = 1.0):
   # check that image is grayscale
    assert image.ndim == 2, 'image should be grayscale'

    dx, dy = canny.sobel_gradients(image)

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

    return xs, ys, scores

"""
  Extract Features - function to compute feature descriptor vectors for each set of interest points

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

def extract_features(image, xs, ys, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   N = len(xs) #should be the same as len(ys)

   window = 3 #not using scale, as its optional

   feats = []
   orients = []

   mag, theta = canny.canny_nmax(image)
   dx, dy = canny.sobel_gradients(image)
   for i in range(N):
    bins = []

    dx_start = xs[i] - 1
    dx_end = xs[i] + 2  if xs[i] + 2 < len(dx) else len(dx) - 1
    dy_start = ys[i] - 1
    dy_end = ys[i] + 2 if ys[i] + 2 < len(dx[0]) else len(dx[0]) - 1
    dir_img_i = dir_img(dx[np.ix_([dx_start, dx_end],[dy_start, dy_end])],
      dy[np.ix_([dx_start, dx_end],[dy_start, dy_end])])

    #print(dir_img_i)
    orients.append(dir_img_i)

    for x in range(3):
      for y in range(3):
        binx = (x+3)/((3+3)/3)
        biny = (y+3)/((3+3)/3)

        bin = np.zeros(8)

        for a in range(window):
          for b in range(window):
            x_disp = int(xs[i] - binx - 1 + a)
            y_disp = int(ys[i] - biny - 1 + b)

            val = int((theta[x_disp,y_disp] - dir_img_i + np.pi)/((np.pi + np.pi)/8)) - 1

            bin[val] += 1
        bins.extend(bin)
    feats.append(bins)

   feats = np.asarray(feats)
   ##########################################################################
   return feats, (np.average(orients) * 180.0/np.pi)

# compute averge orientation at each pixel (given an dy, dx patch)
def dir_img(dx, dy):
  dir_dxys = np.zeros_like(dx)

  for x in range(len(dx)):
    for y in range(len(dx[x])):
      dir_dxys[x][y] = np.arctan2(dy[x][y], dx[x][y]) % np.pi

  orient_xy =  np.average(dir_dxys)

  return orient_xy

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

"""
   Feature Matching - a function to abstract bbf kd trees and nearest neighbor interpolation for matching.

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
def match_features(feats0, feats1, scores0, scores1):
    matches = []
    scores = []

    kdTree = make_kd_tree(feats1)
    for feat0 in feats0:
        matched = nearestNeighbor(feat0, kdTree, feats1)
        matches.append(matched.index)
    matches = np.array(matches, np.int)
    return matches, None

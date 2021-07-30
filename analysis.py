"""
Defines the functions used to track a single
object in several mask.
"""
import numpy as np
import skimage.measure as msr
from .mathtools import nearest_pairs_haversine
from .trajectory import Trajectory


def detect_objects(mask):
    """
    Detects the segmented storms on a segmentation masks, and
    returns numerous properties about each.
    :param ndarray mask: Segmentation mask of shape (H, W);
    :return: a list of properties for each detected object.
        The properties are those of skimage.measure.regionprops
        (See skimage doc).
    """
    # Creates a binary version of the mask: 1 for both VCyc and
    # VMax, 0 for empty pixels
    binary_mask = mask.copy()
    binary_mask[binary_mask != 0] = 1

    # Labelizes the mask, then computes the properties
    labeled_mask = msr.label(binary_mask)
    return msr.regionprops(labeled_mask)


def are_same_storm(obj_1, obj_2):
    """
    Determines whether two segmented storms should be
    considered as the same at two successive points in time.
    :param skimage.measure.RegionProperties obj_1: properties
        for object 1 as returned by skimage.measure.regionprops();
    :param skimage.measure.RegionProperties obj_2: properties
        for object 2.
    """
    # TODO
    return True


def track_trajectories(sequence):
    """
    Tracks the segmented storms accross successive
    segmentation masks.
    :param TSSequence sequence: Sequence object containing the masks
        and associated dates;
    :return: a tuple (T, I) where:
        - T is a list of Trajectory objects. Each trajectory follows
            a single storm.
        - I is a 2D array of shape (Number of masks, Number of trajectories)
            such that I[k, n] is 1 if storm n exists in mask k, and 0
            otherwise.
    """
    # Successive associations algorithm:
    # Initialize the following lists to empty:
    # - U: list of trajectories
    # - A: list of the indexes of active storms in U
    # - C: list of indexes of current trajectories, initially empty
    # - T: list of finished trajectories, initially empty
    # - I: list of indexes of finished trajectories, initially empty
    #
    # For mask k from 0 to the length of the sequence,
    #   Compute the objects O of the current mask;
    #   Let S be the list of active storms [U[a] for a in A]
    #   Find the indexes of the closest pairs P
    #       between objects in S and objects in O
    #       (Pair (0, 1) means object 0 of S corresponds to object O of S);
    #   create an empty list F of finished storms;
    #   For all objects O[j] in O:
    #       Let found_pair be False:
    #       If there is a pair (i, j) in P:
    #           If S[i] could be the same storm:
    #               append O[j] to U[A[i]];
    #               append k to C[A[i]];
    #               found_pair becomes True;
    #           Else:
    #               append i to F;
    #       If not found_pair:
    #           Let o be the object O[j]:
    #           append the singleton [o] to U;
    #           Let n be the new length of U:
    #           append the singleton [n-1] to A
    #           append the singleton [k] to C;
    #   For all indexes i from 0 to the length of S:
    #       if there is no pair (i, something) in P, append i to F;
    #   For all indexes i in F:
    #       add the list U[i] to T;
    #       add the list C[i] to I;
    #       remove the element of value i from A;
    unfinished, active_ind, current_ind, trajs, indexes = [], [], [], [], []
    for k, mask in enumerate(sequence.masks()):
        # Storm objects detected in this mask
        objects = detect_objects(mask)
        # Currently active storm objects
        storms = [unfinished[a] for a in active_ind]

        # Latitudes and longitudes of the object centers
        lats_a = [int(np.round(s.centroid[0])) for s in storms]
        long_a = [int(np.round(s.centroid[1])) for s in storms]
        lats_b = [int(np.round(o.centroid[0])) for o in objects]
        long_b = [int(np.round(o.centroid[1])) for o in objects]
        pairs, distances = nearest_pairs_haversine(lats_a, long_a, lats_b,
                                                   long_b)
        finished = []

        for j, obj in enumerate(objects):
            found_pair = False
            # Browse all pairs trying to find j in one of them
            for ind_s, ind_o in pairs:  # ind_s is i in the algo
                if ind_o == j:  # We found a pair including this object
                    if are_same_storm(obj, storms[ind_s]):
                        unfinished[active_ind[ind_s]].append(obj)
                        current_ind[active_ind[ind_s]].append(k)
                        found_pair = True


def match_object(mask_a, mask_b, latitudes, longitudes):
    """
    Takes two temporally successive mask and matches the storms
    appearing in both mask.
    :param mask_a: Segmentation mask of shape (H, W)
    :param mask_b: Segmentation mask following mask_a in time
    :param latitudes: array giving the latitude for each pixel
    :param longitudes: array giving the longitude for each pixel
    :return: a list L such that L[i] = (Pa, Pb) where Pa and Pb
             are the properties given by skimage.measure.regionprops
             and correspond to the same storm.
    """
    # Creates a a binary version of the mask: 1 for both VCyc and VMax,
    # 0 for empty pixels
    binary_mask_a, binary_mask_b = mask_a.copy(), mask_b.copy()
    binary_mask_a[binary_mask_a != 0] = 1
    binary_mask_b[binary_mask_b != 0] = 1

    # First, compute the properties for all objects
    labels_a, labels_b = msr.label(binary_mask_a), msr.label(binary_mask_b)
    rprops_a = msr.regionprops(labels_a)
    rprops_b = msr.regionprops(labels_b)

    # Now we'll compute the coordinates of the center of each object
    centers_a = [r.centroid for r in rprops_a]
    centers_b = [r.centroid for r in rprops_b]
    # Watchout, the centers are float values and need to be rounded
    lats_a = [latitudes[int(np.round(c[1]))] for c in centers_a]
    longs_a = [longitudes[int(np.round(c[0]))] for c in centers_a]
    lats_b = [latitudes[int(np.round(c[1]))] for c in centers_b]
    longs_b = [longitudes[int(np.round(c[1]))] for c in centers_b]

    # Computes the distance between each objects and returns
    # the closest pairs, and the distance between them.
    # There can remain unmatched objects if the number of objects
    # are inequal.
    nearest_pairs, distances = nearest_pairs_haversine(lats_a, longs_a, lats_b,
                                                       longs_b)
    matched_storms = []
    for i, j in nearest_pairs:
        matched_storms.append(rprops_a[i], rprops_b[j])
    return matched_storms

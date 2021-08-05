"""
Defines the functions used to track a single
object in several mask.
"""
import numpy as np
import skimage.measure as msr
from .mathtools import nearest_pairs_haversine
from .cyclone_object import CycloneObject


def detect_objects(mask):
    """
    Detects the segmented storms on a segmentation mask, and
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
    return filter_objects(msr.regionprops(labeled_mask))


def filter_objects(objects):
    """
    Filters a list of objects to remove the tiniest ones.
    :param objects: list of skimage.measure.RegionProperties objects as
        returned by skimage.measure.regionprops() on a full mask.
    """
    # Removes all objects whose area in pixels is less than a threshold
    return [obj for obj in objects if obj.area >= 50]


def detect_single_object(mask):
    """
    Detects a single segmented cyclone in a segmentation mask,
    and returns the associated RegionProperties object. Should
    not be applied to masks where several cyclones of similar
    dimensions appear.
    :param ndarray mask: Segmentation mask of shape (H, W).
    :return: a RegionProperties object holding information about
        the object.

    The object with largest area out of all is returned.
    All segmentation classes are considered as one class for the
    comparison of areas.
    """
    # Creates a binary version of the mask: 1 for both VCyc and
    # VMax, 0 for empty pixels
    binary_mask = mask.copy()
    binary_mask[binary_mask != 0] = 1
    # Labelizes the mask, then computes the properties
    labeled_mask = msr.label(binary_mask)
    cyclones = msr.regionprops(labeled_mask)

    # Determines the index of the object with the largest area
    # and returns it
    areas = [c.area for c in cyclones]
    largest = np.argmax(areas)
    return cyclones[largest]


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

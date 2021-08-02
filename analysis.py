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
    return filter_objects(msr.regionprops(labeled_mask))


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


def filter_objects(objects):
    """
    Filters a list of objects to remove the tiniest ones.
    :param objects: list of skimage.measure.RegionProperties objects as
        returned by skimage.measure.regionprops() on a full mask.
    """
    # Removes all objects whose area in pixels is less than a threshold
    return [obj for obj in objects if obj.area >= 50]


def track_trajectories(sequence, latitudes, longitudes):
    """
    Tracks the segmented storms accross successive
    segmentation masks.
    :param TSSequence sequence: Sequence object containing the masks
        and associated dates;
    :param latitudes: List or array giving the latitude at each row;
    :param longitudes: List or array giving the longitude at each column;
    :return: a tuple (T, I) where:
        - T is a list of Trajectory objects. Each trajectory follows
            a single storm.
        - I is a 2D array of shape (Number of masks, Number of trajectories)
            such that I[k, n] is 1 if storm n exists in mask k, and 0
            otherwise.
    """
    # A trajectory will be used represented by a list
    # of length the number the masks, and of values either a
    # RegionProperties object or None to indicate that this
    # trajectory did not exist yet in this mask or has already disappeared.
    # For every mask M:
    #   detect the objects in the mask
    #   look at the last objects in each current trajectory
    #   Compute the nearest pairs between the objects in M and those already
    #       in trajectories
    #   If they correspond, at the objects of M into the corresponding
    #       trajectories
    #   For all trajectories whose last element was None or an object
    #       which could not be matched with a new one, add None
    trajectories = []  # List of trajectories
    for k, mask in enumerate(sequence.masks()):
        new_objs = detect_objects(mask)
        old_objs = [traj[-1] for traj in trajectories if traj[-1] is not None]
        # Will keep track of which new objects have been successfully matched
        matched = []

        # If there is no trajectory currently ongoing, we cannot
        # look to match the new objects
        if old_objs != []:
            # List associating the indexes in old_objs with their index in
            # trajectories[]
            traj_indexes = [i for i, traj in enumerate(trajectories) if traj[-1] is not None]

            # Computation of the nearest pairs: We need all objects's coordinates
            lats_old = [latitudes[int(np.round(o.centroid[0]))] for o in old_objs]
            long_old = [longitudes[int(np.round(o.centroid[1]))] for o in old_objs]
            lats_new = [latitudes[int(np.round(o.centroid[0]))] for o in new_objs]
            long_new = [longitudes[int(np.round(o.centroid[1]))] for o in new_objs]
            pairs, distances = nearest_pairs_haversine(lats_old, long_old, lats_new, long_new)

            # For each pair, check that it actually looks like the same storm
            # and if it does, add the new obj to the old one's trajectory
            for ind_old, ind_new in pairs:
                if are_same_storm(old_objs[ind_old], new_objs[ind_new]):
                    trajectories[traj_indexes[ind_old]].append(new_objs[ind_new])
                    # Indicate that the new object has been matched
                    matched.append(ind_new)

            # The trajectories which haven't gained an element during this
            # iteration yet are those whose last object was None (traj is finished
            # already) or did not match any new object (traj has just ended)
            # For all these trajs, we add another None since their associated storm
            # does not appear in the current mask
            for traj in trajectories:
                if len(traj) == k:
                    traj.append(None)

        # For all new objects that were not matched with any already
        # existing objects, create a new trajectory. The new trajectory
        # will contain None for all masks before this one.
        matched = sorted(matched)
        for i, obj in enumerate(new_objs):
            if i not in matched:
                trajectories.append([None for _ in range(k)] + [obj])

    # Compute the presence matrix (See function doc)
    # For all trajs, browse the objects
    # If None then the matrix at this mask and traj is 0, else 1
    presence = np.zeros((len(sequence.masks()), len(trajectories)))
    for i_traj, traj in enumerate(trajectories):
        for i_mask, obj in enumerate(traj):
            presence[i_mask, i_traj] = obj is not None

    # Remove the None at the end
    # Convert to Trajectory objects (the Trajectory class will ignore the None values)
    trajectories = [Trajectory(traj, sequence.masks()) for traj in trajectories]

    return trajectories, presence


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

"""
Defines the TSTracker which is the class used to track a storm
throughout a sequence of segmentations.
"""
from .sequence import TSSequence
from .analysis import track_trajectories


class TSTracker():
    """
    A Tropical Storm Tracker is used to track a depression
    throughout a sequence of segmentation masks. A segmentation
    mask is an array of shape (Height, Width) (which can be viewed
    as a grayscale image), which contains values 0, 1 and 2 for
    classes empty, Max winds and Cyclonic winds respectively.
    """
    def __init__(self, masks, dates, latitudes, longitudes):
        """
        Creates a tracker for a given sequence of segmentation masks.
        :param masks: List or array of storm segmentation masks. These can
                      contain values 0, 1 (max winds area) and 2 (cyclonic
                      winds area).
        :param dates: List of epygram.base.FieldValidity objects defining
                      the validity, basis and term for each mask in masks.
        :param latitudes: Array containing the latitude for each row in
                          the masks.
        :param longitudes: Array giving the longitude for each column
                           in the masks
        """
        self._sequence = TSSequence(masks, dates)
        self._trajectories, self._traj_mask_indexes = track_trajectories(
            self._sequence)

    def masks(self):
        """
        Returns this tracker's segmentation masks.
        """
        return self._sequence.masks()

    def labeled_masks(self):
        """
        Returns the masks where the storms are labelled
        """
        # TODO
        pass

    def dates(self):
        """
        Returns this tracker's FieldValidity dates.
        """
        return self._sequence.dates()

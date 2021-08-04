"""
Defines the TSTracker which is the class used to track a storm
throughout a sequence of segmentations.
"""
import numpy as np
import os
from copy import deepcopy
from .plot import plot_trajectory
from .sequence import TSSequence
from .analysis import track_trajectories


class TSTracker:
    """
    A Tropical Storm Tracker is used to track a depression
    throughout a sequence of segmentation masks. A segmentation
    mask is an array of shape (Height, Width) (which can be viewed
    as a grayscale image), which contains values 0, 1 and 2 for
    classes empty, Max winds and Cyclonic winds respectively.
    """
    def __init__(self):
        """
        Creates a Tropical Storm Tracker to which trajectories can be
        added later.
        """
        self._trajectories = []

    def add_sequence(self, masks, validities, latitudes, longitudes):
        """
        From a sequence of masks and associated validities,
        detects the trajectories and adds them to the tracker.
        :param masks: List or array of storm segmentation masks. These can
                      contain values 0, 1 (max winds area) and 2 (cyclonic
                      winds area).
        :param validities: List of epygram.base.FieldValidity objects defining
                      the validity, basis and term for each mask in masks.
        :param latitudes: Array containing the latitude for each row in
                          the masks.
        :param longitudes: Array giving the longitude for each column
                           in the masks
        """
        self._sequence = TSSequence(masks, validities)
        self._trajectories += track_trajectories(self._sequence, latitudes,
                                                 longitudes)

    def plot_trajectories(self, background=None, dest_dir="."):
        """
        For all trajectories, displays the successive masks of the tracked
        cyclone in a single image.
        :param background: Optional grayscale image of same dimensions
            as the masks. If specified, the trajectories will be showed
            over this image.
        :param dest_dir: Destination directory into which the image is saved.
        """
        if background is None:
            background = np.full_like(self.masks()[0], 255)
        for k, traj in enumerate(self._trajectories):
            plot_trajectory(
                traj, background,
                os.path.join(dest_dir, "trajectory_{}.png".format(k)))

    def _cyclone_is_in_mask(self, index_traj, index_mask):
        """
        Returns True if the cyclone of index index_traj in the trajectories
        list does appear in the mask of index_mask in the sequence.
        """
        return self._presence_matrix[index_mask, index_traj] == 1

    def dates(self):
        """
        Returns this tracker's FieldValidity dates.
        """
        return self._sequence.dates()

    def masks(self):
        """
        Returns this tracker's segmentation masks.
        """
        return self._sequence.masks()

    def nb_masks(self):
        """
        Returns the number of masks used by this tracker.
        """
        return len(self._sequence.masks())

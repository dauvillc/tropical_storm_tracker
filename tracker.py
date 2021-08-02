"""
Defines the TSTracker which is the class used to track a storm
throughout a sequence of segmentations.
"""
import numpy as np
import os
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
        self._trajectories, self._presence_matrix = track_trajectories(
            self._sequence, latitudes, longitudes)

    def labelled_trajectories(self):
        """
        Returned a version of the masks where the cyclones are labelled.
        :return: An array of shape (N, H, W). Each cyclone is given a unique
        label (non-null positive integer) for all masks it appears in.
        Empty pixels are given the value 0.
        """
        # We'll start with blank masks, then paint the storms
        # at each point in its trajectory
        result = np.zeros_like(self.masks())
        for ind_traj, traj in enumerate(self._trajectories):
            # Label for the cyclone followed in this traj (Cannot start at 0 !)
            label = ind_traj + 1
            for ind_mask in range(self.nb_masks()):
                # Check that the cyclone followed in the trajectory ind_traj
                # appears in mask ind_mask
                if self._cyclone_is_in_mask(ind_traj, ind_mask):
                    cyclone = traj[ind_mask]
                    # The following lines paint the pixels in results that are part
                    # of the cyclone with value ind_traj + 1
                    minr, minc, maxr, maxc = cyclone.bbox
                    binary_obj_mask = cyclone.image
                    result[ind_mask, minr:maxr, minc:maxc][binary_obj_mask] = label
        return result

    def plot_trajectories(self, background=None, dest_dir="."):
        """
        For all trajectories, displays the successive masks of the tracked
        cyclone in a single image.
        :param background: Optional grayscale image of shape (masks_H, masks_W).
            If specified, the trajectories will be showed over this image.
        :param dest_dir: Destination directory into which the image is saved.
        """
        if background is None:
            background = np.full_like(self.masks()[0], 255)
        for k, traj in enumerate(self._trajectories):
            plot_trajectory(traj, background, os.path.join(dest_dir, "trajectory_{}.png".format(k)))

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
"""
Defines the TSTracker which is the class used to track a storm
throughout a sequence of segmentations.
"""
import numpy as np
import os
from copy import deepcopy
from epygram.base import FieldValidityList
from .trajectory import Trajectory
from .plot import plot_trajectory


class TSTracker:
    """
    A Tropical Storm Tracker is used to track a depression
    throughout a sequence of segmentation masks. A segmentation
    mask is an array of shape (Height, Width) (which can be viewed
    as a grayscale image), which contains values 0, 1 and 2 for
    classes empty, Max winds and Cyclonic winds respectively.
    """
    def __init__(self, validities: FieldValidityList, latitudes, longitudes):
        """
        Creates a Tropical Storm Tracker to which trajectories can be
        added later.
        :param validities: epygram.base.FieldValidityList,
            validities of the trajectories which will be stored
            in this tracker.
        :param ndarray latitudes: 1D array indicating the latitude for each
            row in the masks.
        :param ndarray longitudes: 1D array indicating the longitude for each
            column in the masks.
        """
        self._trajectories = []
        self._validities = deepcopy(validities)
        self._latitudes = latitudes.copy()
        self._longitudes = longitudes.copy()

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

    def validities(self):
        """
        Returns this tracker's FieldValidity dates.
        """
        return deepcopy(self._validities)


class SingleTrajTracker(TSTracker):
    """
    A type of tropical storm tracker adapted to store
    a single trajectory, which can be updated with a new
    state at any time.
    """
    def __init__(self, validities: FieldValidityList, latitudes, longitudes):
        """
        Creates a Tropical Storm Tracker adapted to store a single trajectory,
        which can be updated with a new state at any time.
        :param validities: epygram.base.FieldValidityList,
            validities of the trajectories which will be stored
            in this tracker.
        :param ndarray latitudes: 1D array indicating the latitude for each
            row in the masks.
        :param ndarray longitudes: 1D array indicating the longitude for each
            column in the masks.
        """
        super().__init__(validities, latitudes, longitudes)
        self._traj = None

    def add_new_state(self, mask):
        """
        Detects in a segmentation the segmented object that best continues
        the trajectory of the tracked cyclone and adds it.
        """
        if self._traj is None:
            self._traj = Trajectory(None, self._latitudes, self._longitudes)
            self._trajectories = [self._traj]
        self._traj.add_state(mask, self._validities[self.nb_states()])

    def nb_states(self):
        """
        Returns the number of states currently in the trajectory.
        """
        if self._traj is None:
            return 0
        else:
            return len(self._traj)

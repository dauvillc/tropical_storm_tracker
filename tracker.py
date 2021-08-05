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
from .sequence import TSSequence


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
    def __init__(self, latitudes, longitudes):
        """
        Creates a Tropical Storm Tracker adapted to store a single trajectory,
        which can be updated with a new state at any time.
        :param ndarray latitudes: 1D array indicating the latitude for each
            row in the masks.
        :param ndarray longitudes: 1D array indicating the longitude for each
            column in the masks.
        """
        super().__init__(FieldValidityList(), latitudes, longitudes)
        self._traj = None

    def add_new_state(self, mask, validity):
        """
        Detects in a segmentation the segmented object that best continues
        the trajectory of the tracked cyclone and adds it.
        :param mask: array of shape (H, W); segmentation mask containing
            the segmented cyclone and supposedly continues the trajectory.
        :param FieldValidity validity: validity of the new state.
        """
        if self._traj is None:
            self._traj = Trajectory(None, self._latitudes, self._longitudes)
            self._trajectories = [self._traj]
            self._validities = FieldValidityList([validity])
        else:
            # returns True if the masks actually continued the trajectory
            if self._traj.add_state(mask, self._validities[self.nb_states()]):
                self._validities.extend([validity])

    def nb_states(self):
        """
        Returns the number of states currently in the trajectory.
        """
        if self._traj is None:
            return 0
        else:
            return len(self._traj)


class MultipleTrajTracker(TSTracker):
    """
    A type of tropical storm tracker adapted to store multiple
    trajectories. The trajectories cannot be updated separately,
    though they can be updated all at once.
    """
    def __init__(self, validities, latitudes, longitudes):
        """
        Creates a tropical storm tracker adapted to store multiple
        trajectories. The validities must be indicated at object
        creation.
        :param FieldValidityList validities: validities the states
            of the trajectories which will be added to this tracker.
        :param latitudes: 1D array giving the latitude at each row of
            the masks.
        :param longitudes: 1D array giving the longitude at each column
            of the masks.
        """
        super().__init__(validities, latitudes, longitudes)

    def add_trajectory(self, masks):
        """
        Detects the interesting object in each masks and builds
        a trajectory from those objects. Make sure the masks actually
        correspond to this tracker's validities before calling this.
        :param masks: array of shape (N, height, width); segmentation
            masks containing the segmented cyclone at each point in time.
        """
        sequence = TSSequence(masks, self.validities())
        new_traj = Trajectory(sequence, self._latitudes, self._longitudes)
        self._trajectories.append(new_traj)

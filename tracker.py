"""
Defines the TSTracker which is the class used to track a storm
throughout a sequence of segmentations.
"""
import os
from copy import deepcopy
from epygram.base import FieldValidityList
from .trajectory import Trajectory, load_trajectory
from .sequence import TSSequence
from .tools import save_validities, save_coordinates
from .tools import load_validities, load_coordinates
from .plot import TSPlotter


def load_multiple_traj_tracker(source_dir):
    """
    Loads a MultipleTrajTracker from a directory.
    """
    validities = load_validities(os.path.join(source_dir, "validities.txt"))
    lats, longs = load_coordinates(os.path.join(source_dir, "coordinates.txt"))
    tracker = MultipleTrajTracker(validities, lats, longs)

    # The trajectories are stored in the subfolders
    # source_dir/trajectories/traj_i
    for traj_dir in os.listdir(os.path.join(source_dir, "trajectories")):
        traj_dir = os.path.join(source_dir, "trajectories", traj_dir)
        tracker._add_trajectory_no_build(load_trajectory(traj_dir))
    return tracker


def load_single_traj_tracker(source_dir):
    """
    Loads a SingleTrajTracker from a directory.
    """
    lats, longs = load_coordinates(os.path.join(source_dir, "coordinates.txt"))
    tracker = SingleTrajTracker(lats, longs)
    traj_dir = os.path.join(source_dir, "trajectories", "traj_0")
    tracker.set_trajectory(load_trajectory(traj_dir))
    return tracker


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

    def validities(self):
        """
        Returns this tracker's FieldValidity dates.
        """
        return deepcopy(self._validities)

    def save(self, dest_dir):
        """
        Saves this tracker's data (including trajectories)
        to a destination directory.
        :param dest_dir: path to the save directory. Pre-existing
            files might be erased.
        """
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        # Saves the validities and coordinates ranges
        save_validities(self.validities(),
                        os.path.join(dest_dir, "validities.txt"))
        save_coordinates(self._latitudes, self._longitudes,
                         os.path.join(dest_dir, "coordinates.txt"))
        # Trajectories are saved in dest_dir/trajectories
        trajs_dir = os.path.join(dest_dir, "trajectories")
        if not os.path.exists(trajs_dir):
            os.makedirs(trajs_dir)
        for k, traj in enumerate(self._trajectories):
            traj.save(os.path.join(trajs_dir, "traj_{}".format(k)))


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
        self._start_new_traj()
        # The SingleTrajTracker has a current traj and ended trajectories
        # When the current trajectory seems to end (i.e. nothing is detected on
        # a mask), it is classified as ended, and stored in this list.
        self._ended_trajs = []
        self._never_used = True
        self._domain_height = latitudes.shape[0]
        self._domain_width = longitudes.shape[0]

    def add_new_state(self, mask, validity, ff10m_field=None):
        """
        Detects in a segmentation the segmented object that best continues
        the trajectory of the tracked cyclone and adds it.
        :param mask: array of shape (H, W); segmentation mask containing
            the segmented cyclone and supposedly continues the trajectory.
        :param FieldValidity validity: validity of the new state.
        :param ff10m_field: array of shape (H, W).
            FF10m wind speed field associated with the segmentation mask.
            The field should be in m/s.
        :return: True if the continuation for the trajectory was found,
            False otherwise.
        """
        # returns True if the masks actually continued the trajectory
        if self._traj.add_state(mask, validity, ff10m_field):
            if self._never_used:
                self._validities = FieldValidityList([validity])
                self._never_used = False
            else:
                self._validities.extend([validity])
            return True
        else:
            # Set the current trajectory as ended if it was not empty
            if not self._traj.empty():
                self._ended_trajs.append(self._traj)
                self._start_new_traj()
            return False

    def plot_current_trajectory(self, to_file):
        """
        Plots this tracker's trajectory and displays various information.
        :param to_file: Image file into which the trajectory is saved.
        """
        lat_range = min(self._latitudes), max(self._latitudes)
        long_range = min(self._longitudes), max(self._longitudes)
        # Creates the plotter object and lets the traj plot itself using it
        plotter = TSPlotter(lat_range, long_range, self._domain_height,
                            self._domain_width)
        self._traj.cartoplot(plotter)
        plotter.save_image(to_file)

    def set_trajectory(self, trajectory):
        """
        Setter function for the _trajectory attribute.
        """
        self._traj = trajectory

    def _start_new_traj(self):
        """
        Initiates a new trajectory.
        """
        self._traj = Trajectory(None, self._latitudes, self._longitudes)
        self._trajectories = [self._traj]

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

    def add_trajectory(self, masks, ff10m_fields=None):
        """
        Detects the interesting object in each masks and builds
        a trajectory from those objects. Make sure the masks actually
        correspond to this tracker's validities before calling this.
        :param masks: array of shape (N, height, width); segmentation
            masks containing the segmented cyclone at each point in time.
        :param ff10m_fields: array of shape (N, height, width): FF10m wind
            fields in m/s associated with the segmentation masks.
        """
        sequence = TSSequence(masks, self.validities(), ff10m_fields)
        new_traj = Trajectory(sequence, self._latitudes, self._longitudes)
        self._trajectories.append(new_traj)

    def _add_trajectory_no_build(self, trajectory):
        """
        Adds an already built Trajectory to the tracker.
        """
        self._trajectories.append(trajectory)

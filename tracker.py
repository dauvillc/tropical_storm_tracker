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
    validities = load_validities(os.path.join(source_dir, "validities.txt"))
    tracker = SingleTrajTracker(lats, longs)

    # Loads the current trajectory (which may be empty)
    traj_dir = os.path.join(source_dir, "current_trajectory")
    current_traj = load_trajectory(traj_dir)

    # Loads the ended trajectories (this is similar to the MultipleTrajTracker)
    ended_trajs = []
    for traj_dir in os.listdir(os.path.join(source_dir, "trajectories")):
        traj_dir = os.path.join(source_dir, "trajectories", traj_dir)
        ended_trajs.append(load_trajectory(traj_dir))

    # Lets the tracker re-build itself correctly
    tracker._load(validities, current_traj, ended_trajs)
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

    def plot_trajectories(self, to_file):
        """
        Plots all trajectories (including ended trajectories) stored in this
        tracker.
        :param to_file: Image file into which the figure is saved.
        """
        # Initializes a Plotter with an empty image
        lat_range, long_range = self.latlon_ranges()
        plotter = TSPlotter(self._latitudes, self._longitudes)
        # Draws each trajectory on the plotter
        if all([t.empty() for t in self._trajectories]):
            plotter.add_central_annotation("No trajectories detected")
        else:
            for traj in self._trajectories:
                traj.display_on_plotter(plotter)
        plotter.save_image(to_file)

    def save(self, dest_dir):
        """
        Saves this tracker's data (including trajectories)
        to a destination directory.
        :param dest_dir: path to the save directory. Pre-existing
            files might be erased.

        The following files / rep are created:
        - validities.txt: Writes a line for each validity
        - coordinates.txt: Gives the lat / long ranges to recreate the grid
        - trajectories: Folder containing the trajectories
        """
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        # Saves the validities and coordinates ranges
        save_validities(self._validities,
                        os.path.join(dest_dir, "validities.txt"))
        save_coordinates(self._latitudes, self._longitudes,
                         os.path.join(dest_dir, "coordinates.txt"))

        # Trajectories are saved in dest_dir/trajectories
        trajs_dir = os.path.join(dest_dir, "trajectories")
        if not os.path.exists(trajs_dir):
            os.makedirs(trajs_dir)
        for k, traj in enumerate(self._trajectories):
            traj.save(os.path.join(trajs_dir, "traj_{}".format(k)))

    def latlon_ranges(self):
        """
        Returns a tuple ((min lat, max lat), (min long, max long)) giving
        the ranges of the geographical coordinates. Minimum values are
        included, while max values are excluded.
        """
        lat_range = min(self._latitudes), max(self._latitudes)
        long_range = min(self._longitudes), max(self._longitudes)
        return lat_range, long_range

    def grid_shape(self):
        """
        Returns this tracker's grid dimensions as a tuple
        (height, width).
        """
        return (self._latitudes.shape[0], self._longitudes.shape[0])

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
        # FieldValidityList() returns [None] instead of [] (Who knows why)
        # so we need to remove the None element
        empty_valid_list = FieldValidityList()
        empty_valid_list.pop(0)
        super().__init__(empty_valid_list, latitudes, longitudes)

        self._traj = None
        self._start_new_traj()
        # The SingleTrajTracker has a current traj and ended trajectories
        # When the current trajectory seems to end (i.e. nothing is detected on
        # a mask), it is classified as ended, and stored in this list.
        self._ended_trajs = []
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
        self._validities.append(validity)
        # returns True if the masks actually continued the trajectory
        if self._traj.add_state(mask, validity, ff10m_field):
            return True
        else:
            # Set the current trajectory as ended if it was not empty
            # and starts a new trajectory
            if not self._traj.empty():
                self._ended_trajs.append(self._traj)
                self._start_new_traj()
            return False

    def plot_current_trajectory(self, to_file):
        """
        Plots this tracker's trajectory and displays various information.
        :param to_file: Image file into which the trajectory is saved.
        """
        if not self.is_initialized():
            raise ValueError(
                "Tried to plot an uninitialized tracker. Try adding\
a new validity to the tracker before plotting its trajectory.")
        lat_range, long_range = self.latlon_ranges()
        # Creates the plotter object and lets the traj use it to plot itself
        plotter = TSPlotter(self._latitudes, self._longitudes)

        # Sets the figure title
        current_val = self.current_validity()
        plotter.set_fig_title(
            current_val.get().strftime("Trajectory - %Y-%m-%d-%H"))

        if self._traj.empty():
            plotter.add_central_annotation("No current detection")
        else:
            self._traj.display_on_plotter(plotter)
        plotter.save_image(to_file)

    def evolution_graph(self, to_file):
        """
        Plots this tracker's current trajectory evolution graph.
        :param to_file: Image file into which the figure is saved.
        """
        if self._traj.empty():
            print("Not making the evolution graph: no ongoing trajectory.")
            return
        self._traj.evolution_graph(to_file)

    def plot_trajectories(self, to_file):
        """
        Plots all trajectories (including ended trajectories) stored in this
        tracker.
        :param to_file: Image file into which the figure is saved.
        """
        # We set self._trajectories to [current + ended trajs]
        # so that the upper-class method plots them all
        self._trajectories = [self._traj] + self._ended_trajs
        super().plot_trajectories(to_file)

    def _set_trajectory(self, trajectory):
        """
        Setter function for the _trajectory attribute. Should not be used
        by an external user.
        """
        self._traj = trajectory

    def _start_new_traj(self):
        """
        Initiates a new trajectory.
        """
        self._traj = Trajectory(None, self._latitudes, self._longitudes)

    def save(self, dest_dir):
        """
        Saves this tracker's data (including trajectories)
        to a destination directory.
        :param dest_dir: path to the save directory. Pre-existing
            files might be erased.

        The following files / reps are created:
        - validities.txt: Writes a line for each validity
        - coordinates.txt: Gives the lat / long ranges to recreate the grid
        - trajectories: Folder containing the ended trajectories
        - current_traj: Directory containing the current trajectory
        """
        # We set self._trajectories to self._ended_trajs so that the
        # upper-class save() method saves those under dest_dir/trajectories
        self._trajectories = self._ended_trajs
        super().save(dest_dir)
        # Saves the current trajectory
        self._traj.save(os.path.join(dest_dir, "current_trajectory"))

    def _load(self, validities, current_traj, ended_trajs):
        """
        Rebuilds the tracker from external data (for example, loaded from a
        save directory).
        :param validities: FieldValidityList giving the tracker's validities;
        :param current_trajs: current Trajectory of this tracker;
        :param ended_trajs: list of ended trajectories.
        """
        self._validities = validities
        self._set_trajectory(current_traj)
        self._ended_trajs = ended_trajs

    def nb_states(self):
        """
        Returns the number of states currently in the trajectory.
        """
        if self._traj is None:
            return 0
        else:
            return len(self._traj)

    def current_validity(self):
        """
        Returns the current validity as a FieldValidity object.
        """
        if len(self._validities) == 0:
            return None
        return deepcopy(self._validities[-1])

    def is_initialized(self):
        """
        Returns True if at least one state has been added
        to the tracker (even if nothing was detected). Returns
        False otherwise.
        """
        return self.current_validity() is not None


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

    def add_trajectory(self, masks, ff10m_fields):
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

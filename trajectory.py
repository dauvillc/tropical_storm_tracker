"""
Defines the CycloneObject and Trajectory classes.
"""
import numpy as np
import skimage.measure as msr
import os
import matplotlib.pyplot as plt

# preferrably loads cPickle since it is faster
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from epygram.base import FieldValidity, FieldValidityList
from .sequence import TSSequence, load_sequence
from .mathtools import haversine_distances
from .cyclone_object import CycloneObject
from .plot import TSPlotter, _colors_
from .tools import save_coordinates, load_coordinates


def load_trajectory(source_dir):
    """
    Loads a Trajectory from a source directory.
    """
    latitudes, longitudes = load_coordinates(
        os.path.join(source_dir, "coordinates.txt"))
    sequence = load_sequence(source_dir)
    traj = Trajectory(None, latitudes, longitudes)
    if sequence is None:
        # If the sequence could not be load, it means
        # the trajectory was empty when it was saved.
        # Thus we return an empty traj (The coords are still known however !)
        return traj
    # WATCHOUT: We do not build the trajectory with the direct
    # constructor, as it would recompute the objects
    # We manually set the attributes instead. This is risky and
    # should NEVER be done outside of this method.
    traj._sequence = sequence
    with open(os.path.join(source_dir, "cyclones.obj"), "rb") as cycfile:
        traj._objects = pickle.load(cycfile)
    return traj


class Trajectory:
    """
    A Trajectory contains information about the successive
    states (location, size, mask, ...) of a segmented
    storm object.
    """
    def __init__(self, sequence, latitudes, longitudes):
        """
        Creates a Trajectory for a specific segmented storm.
        :param sequence: TSSequence from which the trajectory is taken
            from, or None to initiate an empty trajectory.
        :param ndarray latitudes: 1D array indicating the latitude for each
            row in the masks, sorted in decreasing order.
        :param ndarray longitudes: 1D array indicating the longitude for each
            column in the masks, sorted in increasing order.
        """
        self._sequence = None
        self._objects = []
        self._latitudes = latitudes
        self._longitudes = longitudes
        # If a sequence was given, builds the trajectory
        # from the successive data tuples
        if sequence is not None:
            masks = sequence.masks()
            ff10ms = sequence.ff10m()
            for mask, val, ff10m in zip(masks, sequence.validities(), ff10ms):
                if not self.add_state(mask, val, ff10m):
                    break

    def add_state(self, mask, validity, ff10m_field):
        """
        Adds a new state to the trajectory.
        :param mask: Array of shape (H, W). Segmentation containing
            the segmented cyclone that is tracked by this trajectory, in its
            next state.
        :param FieldValidity object giving the basis and term of the new
            cyclone state.
        :param ff10m_field: array of shape (H, W). FF10m field in m/s
            associated with the mask and validity.
        :return: True if the next state was found, False otherwise
        """
        # We try to find the right object in the mask to continue
        # the trajectory
        new_cyc = self.match_new_object(mask, validity, ff10m_field)
        if new_cyc is None:
            print("No possible continuation for the trajectory\
 found for validity {}".format(validity.get().strftime("%Y-%m-%d-%H")))
            return False
        # In case we found a continuation,
        # if this traj is empty, we create the sequence
        if self.empty():
            self._sequence = TSSequence([mask], FieldValidityList([validity]),
                                        [ff10m_field])
        else:
            # Else, we just add the new data to the sequence
            self._sequence.add(mask, validity, ff10m_field)
        self._objects.append(new_cyc)
        return True

    def match_new_object(self, mask, validity, ff10m):
        """
        Detects all segmented cyclones in a given segmentation mask,
        and returns the one that best continues this trajectory.
        :param mask: array of shape (H, W); segmentation mask that
            continues this trajectory.
        :param validity: FieldValidity, Validity of the continuation
        :param ff10m: array of shape (H, W); ff10m field associated
            with the mask.

        The matching is done as such:
        1 Find all objects in the new mask;
        2 Take the closest object
        3 If the closest object verifies a size criterion and isn't too
            far away, return it
        4 If the size criterion fails, remove this object and go back to 2.

        If the trajectory is initially empty, the object of largest area
        is chosen as the starting state.
        """
        # We'll need a binary version of the masks:
        # 0 for empty pixels, 1 for both VCyc and VMax
        binary_mask = mask.copy()
        binary_mask[mask != 0] = 1
        # We can now detect the segmented objects
        objs = msr.regionprops(msr.label(binary_mask))
        # Case no object was found (the mask is empty)
        if len(objs) == 0:
            return None

        # Case where the trajectory is empty
        if len(self) == 0:
            areas = [o.area for o in objs]
            return CycloneObject(mask, objs[np.argmax(areas)], validity,
                                 self._latitudes, self._longitudes, ff10m)

        # Center [lat, long] for each object detected
        centers = [(self._latitudes[int(o.centroid[0])],
                    self._longitudes[int(o.centroid[1])]) for o in objs]
        # Distances between the center of the last state in this traj
        # and the objects that were juste detected
        last_state = self.objects()[-1]
        distances = list(haversine_distances(last_state.center, centers))
        # To find the best-matching object, we'll test whether the closest
        # one matches, until we find one that does OR no object checks out.
        while len(distances) > 0:
            closest = np.argmin(distances)
            closest_obj = CycloneObject(mask, objs[closest], validity,
                                        self._latitudes, self._longitudes,
                                        ff10m)
            if last_state.can_be_next_state(closest_obj):
                # We found the continuation object, we add it to the traj
                return closest_obj
            # The object didn't check the criteria, we try the other ones
            distances.pop(closest)
        # If this is reached, then no object matched as a continuation
        return None

    def cartoplot(self, to_file):
        """
        Plots the trajectory as well as information about the cyclone
        at each time step.
        :param to_file: Image file into which the figure is saved.
        """
        # Creates an empty TSPlotter and uses it to render the figure
        lat_range, long_range = self.latlon_ranges()
        plotter = TSPlotter(self._latitudes, self._longitudes)
        self.display_on_plotter(plotter)
        plotter.save_image(to_file)

    def display_on_plotter(self, plotter):
        """
        Adds the trajectory to a TSPlotter's image.
        :param plotter: TSPlotter object used to renderer a figure.
        """
        # We first draw every VCyc area, and THEN the VMax areas
        # since we want the VMax areas to fully appear as they
        # represent more dangerous areas for the populations
        for i, cyc in enumerate(self.objects()):
            plotter.draw_cyclone(cyc, alpha=0.65, seg_class=1)
        # Draws every CycloneObject's VMax area with the plotter
        # also adds the textual information
        for i, cyc in enumerate(self.objects()):
            # The offset between the textual annotations and the cyclones
            # varies between each object, to avoid the annotations
            # overlapping one another
            offx, offy = 0, (60 + (i % 4) * 20) * (-1)**(i & 1)
            # Specifies what info we want to display in the annotations
            text_info = ["max_wind", "term"]
            # We only display the basis on the first state in the traj
            if i == 0:
                text_info += ["basis"]
            plotter.draw_cyclone(cyc,
                                 alpha=0.65,
                                 text_offset=(offx, offy),
                                 text_info=text_info,
                                 seg_class=2)

    def evolution_graph(self, to_file):
        """
        Plots on a single figure several values regarding the evolution
        of the trajectory.
        :param to_file: Image file into which the figure is saved.
        """
        fig, host = plt.subplots(figsize=(8, 5))

        if not self.empty():
            # Figure main title, indicating the basis
            first_val = self._sequence.validities()[0]
            title = "Trajectory " + first_val.getbasis().strftime(
                "%Y-%m-%d-%H")
            fig.suptitle(title)
            # the x-axis ticks are the terms
            terms = [
                "+{:n}".format(t.total_seconds() / 3600)
                for t in self._sequence.validities().term()
            ]
            x_locs = range(len(terms))

            # Colors attribution
            maxwind_color = _colors_[0]
            maxwinddiam_color = _colors_[1]
            maxwindarea_color = _colors_[2]

            # First graph: Max wind speed
            max_winds = [cyc.maxwind for cyc in self.objects()]
            host.plot(x_locs, max_winds, color=maxwind_color)
            host.set_ylabel("1-minute sustained wind speed (m/s)")
            host.set_ylim(0, 80)
            host.yaxis.label.set_color(maxwind_color)

            # Second graph: VMax diameter
            par1 = host.twinx()
            vmax_diams = [cyc.maxwind_diameter for cyc in self.objects()]
            par1.plot(x_locs, vmax_diams, color=maxwinddiam_color)
            par1.set_ylabel("Max wind area diameter (km)")
            par1.yaxis.label.set_color(maxwinddiam_color)
            par1.set_ylim(0, max(vmax_diams) * 1.1)

            # Third graph: VMax area
            par2 = host.twinx()
            vmax_areas = [cyc.maxwind_area for cyc in self.objects()]
            par2.plot(x_locs, vmax_areas, color=maxwindarea_color)
            par2.set_ylabel("Max wind area surface (pixels)")
            par2.yaxis.label.set_color(maxwindarea_color)
            par2.spines["right"].set_position(("outward", 40))
            par2.set_ylim(0, max(vmax_areas) * 1.1)

            # Adds the terms as the X-axis labels
            host.set_xticks(x_locs)
            host.set_xticklabels(terms)
            host.set_xlabel("Term (hours)")
            host.set_xlim(0, len(terms) - 1)
            # Draws vertical lines at each term
            host.grid(axis="x", which="both", linestyle="--")

        fig.savefig(to_file, bbox_inches="tight")
        plt.close(fig)

    def save(self, dest_dir):
        """
        Saves the trajectory to a directory. Any information inside
        that directory is erased.
        The following files are created:
        - coordinates.txt: Gives the lat / long ranges to recreate the grid
        - validities.txt: Writes a line for each validity
        - masks.h5: stores the segmentations as an array of shape (N, H, W)
        - cyclones.obj: stores the CycloneObject list
        - trajectory.png: Plot of the trajectory
        """
        # Erases the directory if it already exists
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        # The saving is done as such:
        # - a file "validites.txt" gives the validities of this traj
        # - a file "masks.h5" stores the original masks, in an array
        #   of shape (N, H, W);
        # - a file "coordinates.txt" stores the lat / long ranges
        #   in the format min:max:step (max is not included).
        # - a file "cyclones.obj" contains the list of the cyclone
        #   objects;
        # - an image "trajectory.png" shows the trajectory on a single map.
        save_coordinates(self._latitudes, self._longitudes,
                         os.path.join(dest_dir, "coordinates.txt"))
        if self.empty():
            # If the trajectory is empty, there's nothing to save
            # other than the coordinates
            return
        self._sequence.save(dest_dir)
        self.cartoplot(os.path.join(dest_dir, "trajectory.png"))
        with open(os.path.join(dest_dir, "cyclones.obj"), "wb") as cfile:
            pickle.dump(self._objects, cfile)

    def validities(self):
        """
        Returns this trajectory's validities as a FieldValidityList
        object.
        """
        if self._sequence is None:
            return None
        return self._sequence.validities()

    def objects(self):
        """
        Returns this trajectory's CycloneObject elements.
        """
        return self._objects

    def grid_shape(self):
        """
        returns a tuple (height, width) of the grid's dimensions.
        """
        if self._sequence is None:
            return None
        return self._sequence.masks()[0].shape

    def latlon_ranges(self):
        """
        Returns a tuple ((min lat, max lat), (min long, max long)) giving
        the ranges of the geographical coordinates. Minimum values are
        included, while max values are excluded.
        """
        lat_range = min(self._latitudes), max(self._latitudes)
        long_range = min(self._longitudes), max(self._longitudes)
        return lat_range, long_range

    def empty(self):
        """
        Returns True if the trajectory is empty.
        """
        return len(self) == 0

    def __getitem__(self, item):
        """
        :param item: Either an integer (index of the cyclone state)
            or FieldValidity object.
        """
        if self._sequence is None:
            return None
        if isinstance(item, int):
            return self._objects[item]
        elif isinstance(item, FieldValidity):
            for k, validity in self.validities():
                if validity.get() == item.get():
                    return self._objects[k]
        else:
            raise KeyError("Key must be an integer or FieldValidity object,\
                    found type {}".format(type(item)))

    def __iter__(self):
        """
        Iterates over (validity, object) tuples.
        """
        return zip(self._validities, self._objects)

    def __len__(self):
        if self._sequence is None:
            return 0
        else:
            return len(self._objects)

"""
Defines the CycloneObject and Trajectory classes.
"""
import numpy as np
import skimage.measure as msr
from epygram.base import FieldValidity, FieldValidityList
from .sequence import TSSequence
from .mathtools import haversine_distances
from .cyclone_object import CycloneObject
from .plot import draw_cyclone_on_image, cartoplot_image


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
            row in the masks.
        :param ndarray longitudes: 1D array indicating the longitude for each
            column in the masks.
        """
        self._sequence = sequence
        self._objects = []
        self._latitudes = latitudes
        self._longitudes = longitudes
        if sequence is not None:
            masks = sequence.masks()
            for mask, val in zip(masks, sequence.validities()):
                if not self.add_state(mask, val):
                    break

    def objects(self):
        """
        Returns this trajectory's CycloneObject elements.
        """
        return self._objects

    def add_state(self, mask, validity):
        """
        Adds a new state to the trajectory.
        :param mask: Array of shape (H, W). Segmentation containing
            the segmented cyclone that is tracked by this trajectory, in its
            next state.
        :param FieldValidity object giving the basis and term of the new
            cyclone state.
        :return: True if the next state was found, False otherwise
        """
        # We try to find the right object in the mask to continue
        # the trajectory
        new_cyc = self.match_new_object(mask, validity)
        if new_cyc is None:
            print("No possible continuation for the trajectory\
 found for validity {}".format(validity.get().strftime("%Y-%m-%d-%H")))
            return False
        # Build the sequence object if it hasn't been already
        if self._sequence is None:
            self._sequence = TSSequence([mask], FieldValidityList([validity]))
        else:
            self._sequence.add(mask, validity)
        self._objects.append(new_cyc)
        return True

    def match_new_object(self, mask, validity):
        """
        Detects all segmented cyclones in a given segmentation mask,
        and returns the one that best continues this trajectory.
        :param mask: array of shape (H, W); segmentation mask that
            continues this trajectory.
        :param validity: FieldValidity, Validity of the continuation

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
                                 self._latitudes, self._longitudes)

        # Center [lat, long] for each object detected
        centers = [(self._latitudes[int(o.centroid[0])],
                    self._longitudes[int(o.centroid[1])]) for o in objs]
        # Distances between the center of the last state in this traj
        # and the objects that were juste detected
        last_state = self.objects()[-1]
        distances = list(haversine_distances(last_state.center, centers))
        while len(distances) > 0:
            closest = np.argmin(distances)
            closest_obj = CycloneObject(mask, objs[closest], validity,
                                        self._latitudes, self._longitudes)
            if last_state.can_be_next_state(closest_obj):
                # We found the continuation object, we add it to the traj
                return closest_obj
            # The object didn't check the criteria, we try the other ones
            distances.pop(closest)
        return None

    def cartoplot(self, to_file):
        """
        Plots the trajectory using Cartopy.
        :param to_file: image file into which the figure is saved.
        """
        lat_range = min(self._latitudes), max(self._latitudes)
        long_range = min(self._longitudes), max(self._longitudes)
        # Starts with a blank mask, then successively draws each state
        image = np.full((*self._sequence.masks()[0].shape, 3), 255)
        for cyc in self.objects():
            draw_cyclone_on_image(image, cyc)
        cartoplot_image(image, lat_range, long_range, to_file)

    def validities(self):
        """
        Returns this trajectory's validities as a FieldValidityList
        object.
        """
        if self._sequence is None:
            return None
        return self._sequence.validities()

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

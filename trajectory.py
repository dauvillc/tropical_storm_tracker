"""
Defines the CycloneObject and Trajectory classes.
"""
import numpy as np
from copy import deepcopy
from epygram.base import FieldValidity


class CycloneObject:
    """
    The CycloneObject class contains detailed information about
    a specific segmentation object coming from a mask.
    This class is strongly based on skimage.measure.RegionProperties,
    but adds some information such as the original pixels corresponding
    to the object itself;
    """
    def __init__(self, origin_mask, properties, validity):
        """
        Creates a Cyclone object/
        :param origin_mask: Array of shape (H, W); Segmentation mask from which
            the object is taken.
        :param properties: skimage.measure.RegionProperties of this
            object returned by regionprops(origin_mask).
        :param validity: FieldValidity object indicating the basis and term
            of this cyclone segmentation.
        """
        # Copies the attributes from properties into self
        # If you ever need to copy another attribute, add it here.
        # (See skimage.measure.regionprops doc for the list of available
        # attributes).
        self.bbox = properties.bbox
        self.image = properties.image
        self.centroid = properties.centroid
        self.area = properties.area

        # Crops the original mask to the bounding box of this object
        minr, minc, maxr, maxc = self.bbox
        mask = origin_mask[minr:maxr, minc:maxc].copy()

        # Erases all pixels that are not part of the object
        mask[np.logical_not(self.image)] = 0

        self.mask = mask.astype(int)
        self._validity = validity

    def validity(self):
        """
        Returns this cyclone segmentation's FieldValidity.
        """
        return deepcopy(self._validity)


class Trajectory:
    """
    A Trajectory contains information about the successive
    states (location, size, mask, ...) of a segmented
    storm object.
    """
    def __init__(self, properties, sequence, complete=True):
        """
        Creates a Trajectory for a specific segmented storm.
        :param list of skimage.measure.RegionProperties properties:
            List of properties returned by skimage.measure.regionproperties
            for the successive states of the storm.
            This list can also contain None values. If a None is encountered,
            then the mask of same index in origin_masks is ignored.
        :param sequence: TSSequence from which the trajectory is taken
            from.
        :param complete: Boolean, indicates whether the trajectory is complete
            (i.e. the last state is the last state before
            the storm disappears).
        """
        masks = sequence.masks()
        validities = sequence.validities()
        self._objects = [
            CycloneObject(m, p, d)
            for m, p, d in zip(masks, properties, validities) if p is not None
        ]
        self._complete = complete
        self._sequence = sequence

    def objects(self):
        """
        Returns this trajectory's CycloneObject elements.
        """
        return self._objects

    def add_state(self, state_properties, mask, validity):
        """
        Adds a new state to the trajectory.
        :param state_properties: skimage.measure.RegionProperties object
            associated to this storm in the new state.
        :param mask: Array of shape (H, W). Segmentation mask from which
            the RegionProperties object is taken.
        :param FieldValidity object giving the basis and term of this
            cyclone state.
        """
        if self.complete:
            raise ValueError("Tried to add a new state to an already\
                    complete trajectory")
        self._objects.append(CycloneObject(mask, state_properties, validity))

    def is_complete(self):
        """
        Returns True if and only if the trajectory is complete.
        """
        return self._complete

    def validities(self):
        """
        Returns this trajectory's validities as a FieldValidityList
        object.
        """
        return self._sequence.validities()

    def __getitem__(self, item):
        """
        :param item: Either an integer (index of the cyclone state)
            or FieldValidity object.
        """
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
        return len(self._objects)

"""
Defines the CycloneObject and Trajectory classes.
"""
import matplotlib.pyplot as plt
import numpy as np
import inspect


class CycloneObject:
    """
    The CycloneObject class contains detailed information about
    a specific segmentation object coming from a mask.
    This class is strongly based on skimage.measure.RegionProperties,
    but adds some information such as the original pixels corresponding
    to the object itself;
    """

    def __init__(self, origin_mask, properties):
        """
        Creates a Cyclone object/
        :param origin_mask: Segmentation mask from which the object is taken from. Array
            of shape (H, W);
        :param properties: skimage.measure.RegionProperties concerning this object returned by
            regionprops(origin_mask).
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


class Trajectory:
    """
    A Trajectory contains information about the successive
    states (location, size, mask, ...) of a segmented
    storm object.
    """

    def __init__(self, properties, origin_masks):
        """
        Creates a Trajectory for a specific segmented storm.
        :param list of skimage.measure.RegionProperties properties:
            List of properties returned by skimage.measure.regionproperties
            for the successive states of the storm.
            This list can also contain None values. If a None is encountered,
            then the mask of same index in origin_masks is ignored.
        :param origin_masks: Array of shape (N, H, W), original masks
            from which the objects are taken from.
        """
        self._objects = [CycloneObject(m, p) for m, p in zip(origin_masks, properties) if p is not None]

    def objects(self):
        """
        Returns this trajectory's CycloneObject elements.
        """
        return self._objects

    def __getitem__(self, item):
        return self._objects[item]

    def __iter__(self):
        return iter(self._objects)

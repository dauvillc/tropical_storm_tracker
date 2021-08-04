"""
Defines the CycloneObject class.
"""
import numpy as np
from copy import deepcopy


class CycloneObject:
    """
    The CycloneObject class contains detailed information about
    a specific segmentation object coming from a mask.
    This class is strongly based on skimage.measure.RegionProperties,
    but adds some information such as the original pixels corresponding
    to the object itself;
    """
    def __init__(self, origin_mask, properties, validity, latitudes,
                 longitudes):
        """
        Creates a Cyclone object/
        :param origin_mask: Array of shape (H, W); Segmentation mask from which
            the object is taken.
        :param properties: skimage.measure.RegionProperties of this
            object returned by regionprops(origin_mask).
        :param validity: FieldValidity object indicating the basis and term
            of this cyclone segmentation.
        :param latitudes: 1D array giving the latitude at each row of the
            masks;
        :param longitudes: 1D array giving the longitude at each column
            of the masks.
        """
        # Copies the attributes from properties into self
        # If you ever need to copy another attribute, add it here.
        # (See skimage.measure.regionprops doc for the list of available
        # attributes).
        self.bbox = properties.bbox
        self.image = properties.image
        self.centroid = properties.centroid
        # Stores the center in geographical coordinates
        self.center = np.array([
            latitudes[int(self.centroid[0])], longitudes[int(self.centroid[1])]
        ])
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

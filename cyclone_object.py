"""
Defines the CycloneObject class.
"""
import numpy as np
from copy import deepcopy
from haversine import haversine


class CycloneObject:
    """
    The CycloneObject class contains detailed information about
    a specific segmentation object coming from a mask.
    This class is strongly based on skimage.measure.RegionProperties,
    but adds some information such as the original pixels corresponding
    to the object itself;
    """
    def __init__(self,
                 origin_mask,
                 properties,
                 validity,
                 latitudes,
                 longitudes,
                 ff10m_field=None):
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
        :param ff10m_field: array of shape (H, W). FF10m field in m/s
            associated with the original mask.
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

        # Crops the wind field to the bounding box of the object and saves it
        self._ff10m = None
        if ff10m_field is not None:
            ff10m_field = ff10m_field[minr:maxr, minc:maxc].copy()
            self._ff10m = ff10m_field

        self.mask = mask.astype(int)
        self._validity = validity

    def can_be_next_state(self, other):
        """
        Tests whether another CyloneObject could be the same storm
        at a forward point in time.
        :param other: other CycloneObject, candidate to be the continuation
            of this.
        """
        # We'll check whether the distance between the two objects
        # is possible. We'll find the time step between the two objects
        # using their validities, then we'll compute their speed since
        # we also have their location. If their speed overcomes a mean 40 km/h,
        # we reject the hypothesis that they are the same storm
        val_self, val_oth = self.validity().get(), other.validity().get()
        # Check that the object is actually forward in time than self
        if val_self >= val_oth:
            return False
        # Time interval in hours
        timedelta = (val_oth - val_self).total_seconds() / 3600
        distance = haversine(self.center, other.center)
        speed = distance / timedelta
        if speed >= 40:
            print(
                "Found an average speed of {:1.1f} km/h, rejecting the object\
 as next step for this cyclone".format(speed))
            return False
        return True

    def validity(self):
        """
        Returns this cyclone segmentation's FieldValidity.
        """
        return deepcopy(self._validity)

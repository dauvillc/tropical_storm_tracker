"""
Defines the Trajectory class.
"""


class Trajectory:
    """
    A Trajectory contains information about the successive
    states (location, size, mask, ...) of a segmented
    storm object.
    """
    def __init__(self, properties):
        """
        Creates a Trajectory for a specific segmented storm.
        :param list of skimage.measure.RegionProperties properties:
            List of properties returned by skimage.measure.regionproperties
            for the successive states of the storm.
        """
        self._properties = properties

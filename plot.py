"""
Defines plotting functions for trajectories and cyclone objects.
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from skimage.io import imsave
from .trajectory import Trajectory

_CLASS_COLORS_ = np.array([[255, 255, 255], [0, 180, 255], [255, 0, 0]])


def plot_trajectory(trajectory, background, to_file=None, mask_alpha=0.3):
    """
    Displays the successive areas of a cyclone in a trajectory on a single
    image.
    :param trajectory: Trajectory object whose object should be plotted;
    :param background: Background image over which the trajectory is showed.
        Should be a grayscale image of shape (H, W). A fully white image
        can be passed to indicate at least the dimensions of the result.
    :param to_file: Optional image filename into which the image should
        be saved.
    :param mask_alpha: Optional float between 0 and 1. Indicates the
        transparency rate for the cyclone areas over the background image.
    :return: The resulting RGB image
    """
    # Makes an RGB copy of background (shape (H, W, 3))
    result = np.repeat(np.expand_dims(background.copy(), axis=-1), 3, axis=-1)

    for cyclone in trajectory.objects():
        minr, minc, maxr, maxc = cyclone.bbox
        cyclone_dest = result[minr:maxr, minc:maxc]
        cyclone_src = _CLASS_COLORS_[cyclone.mask]

        # Boolean array indicating which pixels are part of the cyclone
        cyc_pixels = cyclone.image
        cyclone_dest[cyc_pixels] = (
            (1 - mask_alpha) * cyclone_dest[cyc_pixels] +
            mask_alpha * cyclone_src[cyc_pixels])
    if to_file is not None:
        imsave(to_file, result.astype(np.uint8), check_contrast=False)
    return result

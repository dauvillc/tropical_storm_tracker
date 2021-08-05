"""
Defines plotting functions for trajectories and cyclone objects.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from skimage.io import imsave
from .cyclone_object import CycloneObject

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


def colorize_masks(masks):
    """
    Returns an RGB version of a set of segmentation masks,
    adapted for visualization.
    :param masks: array of shape (N, H, W), containing values
        0 (empty pixels), 1 (VCyc) or 2 (VMax).
    :return: an array of shape (N, H, W, 3) of colorized masks.
    """
    # Array of shape (N, H, W, 3) fully white
    result = np.full((*masks.shape, 3), 255, dtype=np.uint8)
    result[masks == 2] = np.array([255, 0, 0])
    result[masks == 1] = np.array([0, 180, 255])
    return result


def draw_cyclone_on_image(image, cyclone: CycloneObject, alpha=0.5):
    """
    Draws a segmented cyclone onto an RGB image.
    :param image: array of shape (H, W, 3). Image to draw upon,
        modified in-place.
    :param cyclone: CycloneObject to draw.
    :param alpha: float between 0 and 1; opacity of the cyclone
        over the image.
    """
    # boolean array of which pixels are inside the cyclone
    binary_mask = cyclone.image
    minr, minc, maxr, maxc = cyclone.bbox
    # Portion of the image cropped to the cyclone's bbox
    cropped = image[minr:maxr, minc:maxc]
    # Converts the pixels of the RGB image into grayscale
    # before we can draw over them
    cropped[binary_mask][:, 0] = np.mean(cropped, axis=2)[binary_mask]
    cropped[binary_mask][:, 1] = np.mean(cropped, axis=2)[binary_mask]
    cropped[binary_mask][:, 2] = np.mean(cropped, axis=2)[binary_mask]
    # Draws the cyclone's mask over the now gray pixels
    cyc_mask = colorize_masks(np.expand_dims(cyclone.mask,
                                             axis=0))[0][binary_mask]
    cropped[binary_mask] = alpha * cyc_mask + (1 -
                                               alpha) * cropped[binary_mask]
    return image


def cartoplot_seg(masks, lat_range, long_range, to_file):
    """
    Plots one or several segmentations over the coastline using Cartopy.
    :param masks: array of shape (H, W) for a single mask, or (N, H, W)
        for multiple masks.
    :param lat_range: (min latitude, max latitude) tuple;
    :param long_range: (min longitude, max longitude) tuple;
    :param to_file: image file into which the figure is saved.
    """
    # If there are several masks, we need to merge them
    if len(masks.shape) == 2:
        image = colorize_masks(np.expand_dims(masks, axis=0))[0]
    else:
        image = np.amin(colorize_masks(masks), axis=0)
    cartoplot_image(image, lat_range, long_range, to_file)


def cartoplot_image(image, lat_range, long_range, to_file):
    """
    Plots an image and the coastlines using cartopy.
    :param image: array of shape (H, W, 3), image to show;
    :param lat_range: tuple (min latitude, max latitude)
    :param long_range: tuple (min longitude, max longitude)
    :param to_file: image file into which the figure is saved
    """
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.imshow(image,
              origin="upper",
              extent=(*long_range, *lat_range),
              transform=ccrs.PlateCarree(),
              alpha=0.6)
    ax.coastlines(resolution="50m", linewidth=1)
    plt.savefig(to_file)
    plt.close()

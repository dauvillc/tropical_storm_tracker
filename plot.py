"""
Defines plotting functions for trajectories and cyclone objects.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from .cyclone_object import CycloneObject
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

_CLASS_COLORS_ = np.array([[255, 255, 255], [0, 180, 255], [255, 0, 0]])


class TSPlotter:
    """
    The TSPlotter class helps plotting the tracker's information.
    The successive states of a trajectory can be added step by step,
    before making the final plot.
    """
    def __init__(self, latitude_range, longitude_range, height, width):
        """
        Creates a TSPlotter object with an empty imaeg over a given
        geographical area.
        :param latitude_range: (min lat, max lat) tuple indicating the
            latitude range of the plotted area;
        :param longitude_range: (min long, max long) tuple;
        :param height: Height in pixels of the plotted image;
        :param width: Width in pixels of the plotted image.
        """
        self._h, self._w = height, width
        self._extent = (*longitude_range, *latitude_range)
        self._image = np.full((height, width, 3), 255)

        self._fig = plt.figure(figsize=(16, 9))
        self._ax = plt.axes(projection=ccrs.PlateCarree())

    def draw_cyclone(self,
                     cyclone: CycloneObject,
                     alpha=0.5,
                     text_offset=(0, 0)):
        """
        Draws a segmented cyclone onto the plotter's image.
        :param cyclone: CycloneObject to draw.
        :param alpha: float between 0 and 1; opacity of the cyclone
            over the image.
        :param text_offset: Offset in plt points between the cyclone
            and its annotation.
        """
        # boolean array of which pixels are inside the cyclone
        binary_mask = cyclone.image
        minr, minc, maxr, maxc = cyclone.bbox
        # Portion of the image cropped to the cyclone's bbox
        cropped = self._image[minr:maxr, minc:maxc]
        # Converts the pixels of the RGB image into grayscale
        # before we can draw over them
        cropped[binary_mask][:, 0] = np.mean(cropped, axis=2)[binary_mask]
        cropped[binary_mask][:, 1] = np.mean(cropped, axis=2)[binary_mask]
        cropped[binary_mask][:, 2] = np.mean(cropped, axis=2)[binary_mask]
        # Draws the cyclone's mask over the now gray pixels
        cyc_mask = colorize_masks(np.expand_dims(cyclone.mask,
                                                 axis=0))[0][binary_mask]
        cropped[binary_mask] = alpha * cyc_mask + (
            1 - alpha) * cropped[binary_mask]

        # Annotates the cyclone with textual information
        # We need to swap latitude and longitude as the standards for
        # plotting are inversed w/ regard to those of the analysis
        center = cyclone.center[1], cyclone.center[0]
        validity = cyclone.validity().get().strftime("%Y-%m-%d-%H")
        term = int(cyclone.validity().term().total_seconds() / 3600)
        self._ax.annotate("{}+{}h Cat {} - {:1.1f}m/s".format(
            validity, term, cyclone.category, cyclone.maxwind),
                          xy=center,
                          xytext=text_offset,
                          textcoords="offset points",
                          arrowprops=dict(arrowstyle="-", linewidth=0.3),
                          fontsize='xx-small')

        return self._image

    def save_image(self, path):
        """
        Saves the plotter's image to a file.
        :path: Image file into which the map is saved.
        """
        self._ax.imshow(self._image,
                        origin="upper",
                        extent=self._extent,
                        transform=ccrs.PlateCarree(),
                        alpha=0.6)
        gl = self._ax.gridlines(draw_labels=True,
                                color="lightgray",
                                linestyle="--")
        gl.ylabels_left = False
        gl.xlabels_top = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        self._ax.coastlines(resolution="50m", linewidth=1)
        self._fig.savefig(path, bbox_inches="tight")


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
    plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.imshow(image,
              origin="upper",
              extent=(*long_range, *lat_range),
              transform=ccrs.PlateCarree(),
              alpha=0.6)
    ax.coastlines(resolution="50m", linewidth=1)
    plt.savefig(to_file)
    plt.close()

"""
Defines plotting functions for trajectories and cyclone objects.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import numpy as np
from .cyclone_object import CycloneObject
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

_CLASS_COLORS_ = np.array([[255, 255, 255], [140, 140, 210], [230, 70, 50]])
_colors_ = ['#785EF0', '#DC267F', '#FE6100', '#FFB000', '#648FFF']


class TSPlotter:
    """
    The TSPlotter class helps plotting the tracker's information.
    The successive states of a trajectory can be added step by step,
    before making the final plot.
    """
    def __init__(self, latitudes, longitudes):
        """
        Creates a TSPlotter object with an empty imaeg over a given
        geographical area.
        :param latitudes: 1D array giving the latitude at each pixel row
        :param longitudes: 1D array giving the longitude at each pixel row
        """
        self._latitudes = latitudes.copy()
        self._longitudes = longitudes.copy()
        lat_range, long_range = self.latlon_ranges()
        self._extent = (*long_range, *lat_range)

        height, width = latitudes.shape[0], longitudes.shape[0]
        self._h, self._w = height, width
        self._image = np.full((height, width, 3), 255)

        self._fig = plt.figure(figsize=(16, 9))
        self._ax = plt.axes(projection=ccrs.PlateCarree())

    def draw_cyclone(self,
                     cyclone: CycloneObject,
                     alpha=0.5,
                     text_offset=(0, 0),
                     text_info=["term"],
                     seg_class=0):
        """
        Draws a segmented cyclone onto the plotter's image.
        :param cyclone: CycloneObject to draw.
        :param alpha: float between 0 and 1; opacity of the cyclone
            over the image.
        :param text_offset: Offset in plt points between the cyclone
            and its annotation.
        :param text_info: List indicating what information should
            be annotated to the cyclone. Possible values in the list are
            "cat", "max_wind", "basis", "term".
        :param seg_class: Optional integer, specifies a segmentation class
            to draw (Both=0, VCyc=1, VMax=2).
        """
        minr, minc, maxr, maxc = cyclone.bbox
        # Portion of the image cropped to the cyclone's bbox
        cropped = self._image[minr:maxr, minc:maxc]

        # RGB Colored version of the cyclone mask
        cyc_mask = cyclone.mask.copy()
        cyc_mask_color = colorize_masks(np.expand_dims(cyc_mask, axis=0))[0]
        # If a seg class was specified, we need to draw the pixels of the
        # right class only
        if seg_class != 0:
            right_class_pix = cyc_mask == seg_class
            cropped[right_class_pix] = cyc_mask_color[right_class_pix]
        else:
            # Be careful not to draw empty pixels
            cropped[cyc_mask != 0] = cyc_mask_color

        # Annotates the cyclone with textual information
        # Since we should avoid annotating a cyclone twice, we only
        # add the annotation if VMax (or both classes) is displayed
        if seg_class == 0 or seg_class == 2:
            info = []
            if "basis" in text_info:
                val = cyclone.validity()
                info.append(val.getbasis().strftime("%Y-%m-%d-%H") + " ")
            if "term" in text_info:
                term = int(cyclone.validity().term().total_seconds() / 3600)
                info.append("+{}h ".format(term))
            if "cat" in text_info:
                info.append("Cat {} ".format(cyclone.category))
            if "max_wind" in text_info:
                info.append("{:1.1f}m/s ".format(cyclone.maxwind))
            info = "-".join(info)

            if text_info:
                # We need to swap latitude and longitude as the standards for
                # plotting are inversed w/ regard to those of the analysis
                center = cyclone.center[1], cyclone.center[0]
                self._ax.annotate(info,
                                  xy=center,
                                  xytext=text_offset,
                                  textcoords="offset pixels",
                                  arrowprops=dict(arrowstyle="-",
                                                  linewidth=0.15),
                                  fontsize=5,
                                  horizontalalignment="center")

        return self._image

    def save_image(self, path):
        """
        Saves the plotter's image to a file.
        :path: Image file into which the map is saved.
        """
        # Add a transparency channel to the image, and make
        # the white pixels transparent so that they don't paint
        # everything white
        transp_mask = np.full((self._h, self._w, 1),
                              255,
                              dtype=self._image.dtype)
        white_pixels = np.all(self._image == 255, axis=2)
        transp_mask[white_pixels, 0] = 0

        self._ax.imshow(np.concatenate((self._image, transp_mask), axis=-1),
                        origin="upper",
                        extent=self._extent,
                        transform=ccrs.PlateCarree(),
                        zorder=0)
        # Adds the grid
        gl = self._ax.gridlines(draw_labels=True,
                                color="lightgray",
                                linestyle="--",
                                linewidth=0.3)
        # Does the formatting, such as writing "18°S" instead
        # of "-18"
        gl.ylabels_left = False
        gl.xlabels_top = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        # Draws the coastlines and paints the oceans in blue
        self._ax.coastlines(resolution="110m", linewidth=1)
        self._ax.add_feature(cartopy.feature.OCEAN, zorder=0)

        self._fig.savefig(path, bbox_inches="tight")
        plt.close(self._fig)

    def latlon_ranges(self):
        """
        Returns a tuple ((min lat, max lat), (min long, max long)) giving
        the ranges of the geographical coordinates. Minimum values are
        included, while max values are excluded.
        """
        lat_range = min(self._latitudes), max(self._latitudes)
        long_range = min(self._longitudes), max(self._longitudes)
        return lat_range, long_range


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
    result[masks == 2] = _CLASS_COLORS_[2]
    result[masks == 1] = _CLASS_COLORS_[1]
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

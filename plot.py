"""
Defines plotting functions for trajectories and cyclone objects.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import cartopy.crs as ccrs
import cartopy
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from .cyclone_object import CycloneObject

_CLASS_COLORS_ = np.array([[255, 255, 255], [255, 255, 0], [230, 70, 50]])
_colors_ = ['#785EF0', '#DC267F', '#FE6100', '#FFB000', '#648FFF']

matplotlib.use("Agg")


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
        self._image = np.full((height, width), 0)

        self._fig = plt.figure(figsize=(16, 9))
        self._ax = plt.axes(projection=ccrs.PlateCarree())

    def add_central_annotation(self, text):
        """
        Adds a central textual annotation on the plotter's image.
        :param text: text to print.

        The text is added at the center of the image and is meant to
        stand out.
        """
        self._ax.text(np.median(self._longitudes),
                      np.median(self._latitudes),
                      text,
                      fontsize="xx-large",
                      ha="center")

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
        # If a seg class was specified, we need to draw the pixels of the
        # right class only
        if seg_class != 0:
            right_class_pix = cyc_mask == seg_class
            cropped[right_class_pix] = cyc_mask[right_class_pix]
        else:
            # Be careful not to draw empty pixels, which would erase
            # all other drawings
            cropped[cyc_mask != 0] = cyc_mask

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
        # Plots the image with the cartopy transform,
        # using the adapted colormap
        self._ax.imshow(self._image,
                        origin="upper",
                        extent=self._extent,
                        transform=ccrs.PlateCarree(),
                        cmap=segmentation_colormap(),
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
        self._ax.coastlines(resolution="50m", linewidth=1)
        self._ax.add_feature(cartopy.feature.NaturalEarthFeature(
            "physical", "ocean", "50m", facecolor="lightblue"),
                             zorder=0)

        self._fig.savefig(path, bbox_inches="tight")
        plt.close(self._fig)

    def set_fig_title(self, title):
        """
        Sets the figure main title.
        """
        self._ax.set_title(title)

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


def segmentation_colormap():
    """
    Returns a matplotlib ListedColormap object which
    can be used to draw segmentation maps. Three colors
    are contained in the colormap:
    transparent, VCyc color, VMax color
    """
    colors = np.ones((3, 4))  # 3 colors, 4 channels
    colors[0, :3] = _CLASS_COLORS_[0] / 255.
    colors[1, :3] = _CLASS_COLORS_[1] / 255.
    colors[2, :3] = _CLASS_COLORS_[2] / 255.
    # Add the full transparency to the first color
    # (Associated with empty pixels)
    colors[0, 3] = 0
    return pltcolors.ListedColormap(colors, name="segmentation_colormap")

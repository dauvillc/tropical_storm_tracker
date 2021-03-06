"""
Defines plotting functions for trajectories and cyclone objects.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import cartopy.crs as ccrs
import cartopy
import numpy as np
from matplotlib.patches import Patch
from matplotlib import cm
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
        self.start_new_figure()

    def start_new_figure(self):
        """
        Resets the plotter's figure and starts with a blank one.
        """
        self._image = np.full((self._h, self._w), 0)

        self._fig = plt.figure(figsize=(16, 9))
        self._ax = plt.axes(projection=ccrs.PlateCarree())
        self._plotted_img = None

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
        self._plotted_img = self._ax.imshow(self._image,
                                            origin="upper",
                                            extent=self._extent,
                                            transform=ccrs.PlateCarree(),
                                            cmap=self.get_colormap(),
                                            zorder=0)
        # Adds the grid
        gl = self._ax.gridlines(draw_labels=True,
                                color="lightgray",
                                linestyle="--",
                                linewidth=0.8)
        # Does the formatting, such as writing "18??S" instead
        # of "-18"
        gl.ylabels_left = False
        gl.xlabels_top = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        self.add_cartopy_features()
        self._fig.savefig(path, bbox_inches="tight")
        plt.close(self._fig)

    def add_cartopy_features(self):
        """
        Adds the cartopy features to the figure, such
        as the coastlines and painting oceans in blue.
        """
        # Draws the coastlines and paints the oceans in blue
        self._ax.coastlines(resolution="50m", linewidth=1)
        self._ax.add_feature(cartopy.feature.NaturalEarthFeature(
            "physical", "ocean", "50m", facecolor="lightblue"),
                             zorder=0)

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

    def get_colormap(self):
        """
        Returns the matplotlib colormap to use for plotting
        the figures.
        """
        return segmentation_colormap()


class TSProbabilisticPlotter(TSPlotter):
    """"
    A variant of the TSPlotter designed to plot several trajectories
    in a single figure, with a probabilistic approach.
    Each pixel in the final figure will correspond to the ratio of the
    total number of trajectories which classify this pixel as part
    of the cyclone at some time.
    """
    def __init__(self, N_trajectories, latitudes, longitudes):
        """
        :param N_trajectories: Number of trajectories which will be
            plotted in the figure.
        """
        super().__init__(latitudes, longitudes)
        self._N = N_trajectories
        # In the base class, the image has shape (H, W) and contains
        # values 0, 1 or 2.
        # This class will contain N such images, one for each plotted
        # trajectory. After all trajectories have been plotted,
        # the N images are summed up and normalized along axis 0 to obtain
        # the final probability map of shape (H, W).
        self._images = np.repeat(np.expand_dims(self._image, axis=0),
                                 self._N,
                                 axis=0)
        # The attribute "_image" will refer to ONE of the elements of
        # self._images's first axis
        # Therefore, the super-class methods will draw on this element
        # when called. The method next_trajectory() allows the user
        # to switch to the next trajectory (i.e. next element) in
        # self._images
        self._image = self._images[0]
        self._img_index = 0

    def next_trajectory(self):
        """
        Switches to the next trajectory.
        """
        self._img_index += 1
        if self._img_index < self._N:
            self._image = self._images[self._img_index]

    def save_image(self, path):
        """
        Saves the plotter's image to a file.
        :path: Image file into which the map is saved.
        """
        # Converts all non-zero pixels in the sub-images
        # to 1 to avoid classes inbalance
        self._images[self._images > 0] = 1
        # Sums up the N trajectory images, and
        # put it into self._image so that the upper-class
        # save_image() method does all the work.
        summed_img = np.sum(self._images, axis=0)
        summed_img = summed_img / self._N
        self._image = summed_img
        super().save_image(path)

    def add_cartopy_features(self):
        """
        Adds the cartopy features to the figure.
        Contrary to the upper-class version, does not
        paint the oceans.
        """
        # Draws the coastlines only
        self._ax.coastlines(resolution="50m", linewidth=1)
        self._ax.add_feature(cartopy.feature.NaturalEarthFeature(
            "physical", "land", "50m", facecolor="ivory"),
                             zorder=0)
        self._fig.colorbar(self._plotted_img,
                           ax=self._ax,
                           label="Probability of cyclone",
                           shrink=0.75,
                           aspect=30)

    def get_colormap(self):
        """
        Returns a colormap adapted to probabilities plotting.
        """
        # Copies the original colormap
        cmap = cm.get_cmap("YlOrRd")
        cmap = cmap(np.arange(cmap.N))
        # Sets the alpha channel for values '0' at null
        cmap[0, -1] = 0
        return pltcolors.ListedColormap(cmap, name="probas_cmap")


class ModelTestPlotter(TSPlotter):
    """
    Type of Plotter adapted to test a detection model.
    Overlays full segmentation over a wind speed field.
    """
    def __init__(self, latitudes, longitudes, wind_field, mask):
        """
        :param latitudes: 1D array giving the latitude at each image row;
        :param longitudes: 1D array giving the longitude at each ima col;
        :param wind_field: (H, W)-shaped array. FF10m wind field which was used
            to perform the segmentation. Unit must be meters per second.
        :param mask: (H, W)-shaped array or None. Segmentation mask to plot,
            with values {0, 1, 2} for empty, vcyc and vmax classes respectively
            If None, then only the wind field is displayed.
        """
        super().__init__(latitudes, longitudes)
        self._wind_field = wind_field
        self._mask = mask

    def save_image(self, path, include_mask):
        """
        Saves the plotter's image to a file.
        :param path: Image file into which the map is saved.
        :param include_mask: Boolean. Whether to display the segmentation
            masks over the wind field.
        """
        # Plots the wind field before, the mask will be plotted
        # by the upper-class save_image() method since it is
        # stored in self._image
        wind_cmap, wind_norm = wind_field_colormap()
        self._field_plot = self._ax.imshow(self._wind_field,
                                           origin="upper",
                                           extent=self._extent,
                                           transform=ccrs.PlateCarree(),
                                           cmap=wind_cmap,
                                           norm=wind_norm,
                                           zorder=0)
        self._include_mask = False
        if self._mask is not None and include_mask:
            self._include_mask = True
            self._image = self._mask
        super().save_image(path)

    def add_cartopy_features(self):
        """
        Adds the cartopy features to the figure.
        """
        # Draws the coastlines only
        self._ax.coastlines(resolution="50m", linewidth=1)
        self._fig.colorbar(self._field_plot,
                           ax=self._ax,
                           label="Wind speed (m/s)",
                           shrink=0.8,
                           aspect=30,
                           extend="max",
                           ticks=self._field_plot.norm.boundaries)
        if self._include_mask:
            self.add_segmentation_legend()

    def add_segmentation_legend(self):
        """
        Adds a legend for the segmentation classes (VMax, VCyc) colors.
        """
        seg_cmap = self.get_colormap()
        elements = [
            Patch(facecolor="white", edgecolor="black", label="No detection"),
            Patch(facecolor=seg_cmap.colors[1], label="Cyclonic Wind"),
            Patch(facecolor=seg_cmap.colors[2], label="Maximum Wind")
        ]
        # Places the legend to the left of the upper-left corner
        self._ax.legend(bbox_to_anchor=([-0.03, 1.]),
                        handles=elements,
                        loc="upper right",
                        borderaxespad=0.)

    def get_colormap(self):
        """
        Returns the matplotlib colormap to use to plot the masks
        over the wind field.
        """
        colors = np.array([[0, 0, 0, 0], [0, 180, 255, 128], [255, 0, 0, 128]
                           ]) / 255.0
        return pltcolors.ListedColormap(colors, name="mask_over_wind")


def wind_field_colormap():
    """
    Returns (C, B) where:
    - C is the plt ListedColormap object to use to plot
        wind fields with discretized colors;
    - B is the BoundaryNorm object to use along with C.
    """
    # Wind speed boundaries for the changes of color
    speed_limits = np.array([0, 18, 22, 26, 30, 33, 43, 49, 58, 70, 100])
    # Color for each wind speed tier (RGBA format)
    colors = np.stack([
        np.array([0, 0, 0, 0]),
        np.array([210, 210, 210, 255]),
        np.array([190, 190, 190, 255]),
        np.array([170, 170, 170, 255]),
        np.array([150, 150, 150, 255]),
        np.array([75, 0, 80, 255]),
        np.array([120, 21, 122, 255]),
        np.array([165, 42, 165, 255]),
        np.array([210, 64, 208, 255]),
        np.array([255, 85, 250, 255]),
    ])
    speed_limits = speed_limits
    colors = colors / 255.0
    cmap = pltcolors.ListedColormap(colors, name="wind_fields")
    norm = pltcolors.BoundaryNorm(speed_limits, cmap.N)
    return cmap, norm


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

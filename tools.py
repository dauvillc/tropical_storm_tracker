"""
Defines useful functions, mainly for saving and loading
various data types.
"""
import datetime as dt
import h5py
import numpy as np
from epygram.base import FieldValidityList, FieldValidity


def load_hdf5_images(path):
    """
    Loads a batch of images saved in the hdf5 format.
    The HDF5 file should contain a single key named "image".
    :param path: path to the hdf5 to read.
    """
    h5file = h5py.File(path, mode="r")
    return h5file["image"][()]  # [()] transforms to np array


def save_hdf5_images(data, dest_path):
    """
    Saves a batch of images in the hdf5 format.
    :param data: batch of images to save.
    :param dest_path: file path and name into which the
        array is saved.
    """
    h5file = h5py.File(dest_path, mode="w")
    h5file.create_dataset("image", data=data, dtype=data.dtype)


def parse_coordinates_range(str_coords):
    """
    Parses geographical coordinates in the format
    min:max:step into a numpy array.
    """
    low, high, step = str_coords.split(":")
    return np.arange(float(low), float(high), float(step))


def write_coordinates_range(coords_array):
    """
    Inverse function of parse_coordinates_range().
    Takes an array of coordinates (latitudes or longitudes)
    obtained through parse_coordinates_range() and returns
    a summarized string.
    """
    if coords_array.shape[0] < 2:
        raise ValueError("Coordinates array should contain at least\
2 values")
    start, end = np.round(coords_array[-1], 3), np.round(coords_array[0], 3)
    step = np.round(coords_array[0] - coords_array[1], 3)
    end += step  # Since the range stops at end - 1
    return "{}:{}:{}".format(start, end, step)


def save_validities(validities, dest_file):
    """
    Saves a FieldValidityList to a destination file.
    :param validities: FieldValidityList object to save
    :param dest_file: Destination file.
    """
    with open(dest_file, "w") as vfile:
        for val in validities:
            basis = val.getbasis().strftime("%Y-%m-%d-%H")
            term = int(val.term().total_seconds() / 3600)
            vfile.write("{}+{}\n".format(basis, term))


def load_validities(source_file):
    """
    Loads a FieldValidityList from a source file.
    """
    validities = []
    with open(source_file, "r") as src:
        # Each line is in format YYYY-MM-DD-HH+HH
        basis, term = src.readline().split("+")
        basis = dt.datetime.strptime(basis, "%Y-%m-%d-%H")
        term = dt.timedelta(hours=int(term))
        validities.append(FieldValidity(basis + term, basis, term))
    return FieldValidityList(validities)


def save_coordinates(latitudes, longitudes, dest_file):
    """
    Saves lat/lon ranges to a destination file.
    :param latitudes: 1D array giving the latitudes coordinates.
    :param longitudes: 1D array giving the longitudes coordinates.
    :param dest_file: Destination file
    """
    with open(dest_file, "w") as cdfile:
        cdfile.write(write_coordinates_range(latitudes) + "\n")
        cdfile.write(write_coordinates_range(longitudes) + "\n")


def load_coordinates(source_file):
    """
    Inverse of save_coordinates().
    Loads a coordinates grid (latitudes, longitudes) from
    a source file.
    :return: a tuple (lats, longs) of 1D arrays giving the latitudes
        and longitudes of a grid.
    """
    with open(source_file, "r") as src:
        latitudes = parse_coordinates_range(src.readline())
        longitudes = parse_coordinates_range(src.readline())
        return latitudes, longitudes

"""
Defines the TSSequence class, which combines
a sequence of successive segmentation masks with their
associated terms.
"""
import numpy as np
import datetime as dt
import os
from copy import deepcopy
from epygram.base import FieldValidity, FieldValidityList
from .tools import save_hdf5_images, save_validities
from .tools import load_hdf5_images, load_validities


def validity_range(basis, terms, time_step=6):
    """
    Creates a FieldValidityList from
    a given basis and a certain number of terms.
    :param basis: datetime.datetime object specifying the basis
                  from which the validities are taken from;
    :param terms: Number of terms to be included
    :param time_step: Time step between each term, in hours.
    """
    result = []
    for i in range(0, terms):
        term = dt.timedelta(hours=i * time_step)
        validity = basis + term
        result.append(FieldValidity(validity, basis, term))
    return FieldValidityList(result)


def load_sequence(source_dir):
    """
    Loads a TSSequence from a source directory.
    """
    masks = load_hdf5_images(os.path.join(source_dir, "masks.h5"))
    validities = load_validities(os.path.join(source_dir, "validities.txt"))
    ff10m = load_hdf5_images(os.path.join(source_dir, "ff10m.h5"))
    return TSSequence(masks, validities, ff10m)


class TSSequence:
    """
    A TSSequence associates a set of successive segmentation masks of tropical
    cyclones with their terms (Date + hour).

    Internally, the masks are stored in a (N, height, width)-shaped
    array.
    """
    def __init__(self, masks, validities, ff10m_fields):
        """
        Creates a Tropical Storm Sequence.
        :param masks: list or array of storm segmentation masks.
                      The masks should be shape (height, width) and can contain
                      values 0 (empty segmentation), 1 (max winds) and 2
                      (cyclonic winds).
        :param validities: List of epygrame.base.FieldValidity objects,
            defining the validity, basis and term for each segmentation mask.
        :param ff10m_fields: list of FF10m wind fields (as 2D arrays).
            Each field corresponds to a mask m in masks and should have the
            same shape.
        """
        # We could use copies for safety,
        # but the masks might be relatively heavy in memory.
        # Instead, all functions that modify the masks will have to make
        # a copy before.
        self._masks = [m for m in masks]
        self._validities = validities
        self._ff10m = [f for f in ff10m_fields]

    def add(self, mask, validity, ff10m_field):
        """
        Adds a new state to this sequence.
        :param mask: array of shape (H, W), new segmentation mask;
        :param ff10m_field: Array of shape (H, W), FF10m field associated
            with the masks.
        :param validity: FieldValidity associated with the mask.
        """
        self._masks.append(mask)
        self._validities.append(validity)
        self._ff10m.append(ff10m_field)

    def save(self, dest_dir):
        """
        Saves the sequence in a destination directory.
        Two files are created / rewritten:
        - validities.txt gives the validities in YYYY-MM-DD-HH+HH format
          (One validity per line);
        - masks.h5 stores the masks in an array of shape (N, H, W);
        - ff10m.h5 stores the FF10m field in an array of shape (N, H, W).
        """
        save_validities(self.validities(),
                        os.path.join(dest_dir, "validities.txt"))
        save_hdf5_images(self.masks(), os.path.join(dest_dir, "masks.h5"))
        ff10m = self.ff10m()
        save_hdf5_images(ff10m, os.path.join(dest_dir, "ff10m.h5"))

    def masks(self):
        """
        Returns the list of this sequence's segmentation masks
        """
        return np.array(self._masks, dtype=int)

    def validities(self):
        """
        Returns the list of this sequence's FieldValidity objects
        """
        return deepcopy(self._validities)

    def ff10m(self):
        """
        Returns the list of this sequence's FF10m fields.
        """
        return np.array(self._ff10m, dtype=float)

    def __iter__(self):
        return zip(self.masks(), self.validities(), self.ff10m())

    def __str__(self):
        return "Tropical storm segmentation sequence of validities " + str(
            self._validities)

"""
Defines the TSSequence class, which combines
a sequence of successive segmentation masks with their
associated terms.
"""
import numpy as np
import datetime as dt
from copy import deepcopy
from epygram.base import FieldValidity, FieldValidityList


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


class TSSequence:
    """
    A TSSequence associates a set of successive segmentation masks of tropical
    cyclones with their terms (Date + hour).

    Internally, the masks are stored in a (N, height, width)-shaped
    array.
    """
    def __init__(self, masks, validities):
        """
        Creates a Tropical Storm Sequence.
        :param masks: list or array of storm segmentation masks.
                      The masks should be shape (height, width) and can contain
                      values 0 (empty segmentation), 1 (max winds) and 2
                      (cyclonic winds).
        :param validities: List of epygrame.base.FieldValidity objects,
            defining the validity, basis and term for each segmentation mask.
        """
        # We could use copies for safety,
        # but the masks might be relatively heavy in memory.
        # Instead, all functions that modify the masks will have to make
        # a copy before.
        self._masks = np.array([m for m in masks])
        self._validities = validities

    def masks(self):
        """
        Returns the list of this sequence's segmentation masks
        """
        return self._masks

    def validities(self):
        """
        Returns the list of this sequence's FieldValidity objects
        """
        return deepcopy(self._validities)

    def __str__(self):
        return "Tropical storm segmentation sequence of validities " + str(
            self._validities)

"""
Defines the TSSequence class, which combines
a sequence of successive segmentation masks with their
associated terms.
"""
import numpy as np
import datetime as dt
from epygram.base import FieldValidity, FieldValidityList


def validity_range(basis, terms, timestep=6):
    """
    Creates a FieldValidityList from
    a given basis and a certain number of terms.
    :param basis: datetime.datetime object specifying the basis
                  from which the validities are taken from;
    :param terms: Number of terms to be included
    :param timesteps: Time step between each term, in hours.
    """
    result = []
    for i in range(0, terms):
        term = dt.timedelta(hours=i * timestep)
        validity = basis + term
        result.append(FieldValidity(validity, basis, term))
    return FieldValidityList(result)


class TSSequence():
    """
    A TSSequence associates a set of successive segmentation masks of tropical
    cyclones with their terms (Date + hour).

    Internally, the masks are stored in a (N, height, width)-shaped
    array.
    """
    def __init__(self, masks, dates):
        """
        Creates a Tropical Storm Sequence.
        :param masks: list or array of storm segmentation masks.
                      The masks should be shape (height, width) and can contain
                      values 0 (empty segmentation), 1 (max winds) and 2
                      (cyclonic winds).
        :param dates: List of epygrame.base.FieldValidity objects, defining
                      the validity, basis and term for each segmentation mask.
        """
        self._masks = np.array([m.copy() for m in masks])
        self._dates = FieldValidityList([d for d in dates])

    def masks(self):
        """
        Returns the list of this sequence's segmentation masks
        """
        return [m.copy() for m in self._masks]

    def dates(self):
        """
        Returns the list of this sequence's FieldValidity objects
        """
        return self._dates

    def __str__(self):
        return "Tropical storm segmentation sequence of dates " + str(
            self._dates)

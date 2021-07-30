"""
Defines some mathematical functions
"""
import numpy as np
from haversine import haversine_vector


def nearest_pairs_haversine(lats_A, longs_A, lats_B, longs_B):
    """
    takes lists of points coordinates and returns (L, D) where
    - L is a list of pairs such that:
    -- L[0] = (i_A, i_B) such that ||i_A, i_B|| is the smallest possible
    -- L[1] = (j_A, j_B) such that ||j_A, j_B|| is the second smallest
    - ...
    - D is a list such that D[0] == ||i_A, i_B|| (associated distances)
    if there are more points in A than B or inversely, the
    remaining points will be ignored.
    lats and longs are the latitudes and longitudes of points, and
    all distances are computed using the haversine distance.
    """
    assert len(lats_A) == len(longs_A) and len(lats_B) == len(
        longs_B), "missing coordinates"
    points_A = [(lat, longt) for lat, longt in zip(lats_A, longs_A)]
    points_B = [(lat, longt) for lat, longt in zip(lats_B, longs_B)]

    # Computes the distances between all points
    distances = haversine_vector(points_A, points_B, comb=True).T

    nb_pairs = min(len(lats_A), len(lats_B))
    nearest_pairs, smallest_distances = [], []
    for i in range(nb_pairs):
        index_flattened = np.argmin(distances)
        j = index_flattened // distances.shape[1]
        k = index_flattened % distances.shape[1]
        smallest_distances.append(distances[j, k])
        nearest_pairs.append((j, k))

        # Change the distance to infinity
        # so that it's not the min anymor
        distances[j, k] = np.inf

    return nearest_pairs, smallest_distances

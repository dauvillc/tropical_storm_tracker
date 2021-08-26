"""
Defines some mathematical functions
"""
import numpy as np
from haversine import haversine_vector, haversine
from skimage.measure import find_contours


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


def haversine_distances(ref_point, points):
    """
    Returns the Haversine distances between a reference point
    and a list of other points.
    :param ref_point: Reference point, tuple (lat, long)
    :param points: iterable of tuples (lat, long)
    """
    distances = np.empty((len(points), ))
    for k, point in enumerate(points):
        distances[k] = haversine(ref_point, point)
    return distances


def polygon_haversine_diameter(vertices):
    """
    Computes the diameter of a polygon using the
    haversine distance.
    :param vertices: (N_vertices, 2) np array giving
        the coordinates of the vertices. Columns correspond to the
        longitudes / latitudes of each vertex.
    :return: the diameter (max distance between two vertices)
        using the haversine distance.
    """
    distances = haversine_vector(vertices, vertices, comb=True)
    return np.max(distances)


def mask_haversine_diameter(mask, latitudes, longitudes):
    """
    Computes the diameter of a segmented object, using the
    haversine distance.
    :param mask: (H, W) binary array giving the segmented object.
        All non-zero pixels will be considered as part of the object.
    :param latitudes: 1D array giving the latitude at each column of the mask.
    :param longitudes: 1D array giving the longitude at each row of the mask.
    :return: The diameter (max distance between two points of the object)
        computed using the haversine distance.
    """
    # gets the polygon vertices as an array of shape (n_vertices, 2)
    vertices = find_contours(mask, 0.5)
    if len(vertices) == 0:
        return np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
    vertices = vertices[0].astype(int)
    # Transforms it into an array of shape (n_vertices, 2) where column 0
    # is the latitudes coords, and 1 is the longitudes
    vertices_coords = np.stack(
        [latitudes[vertices[:, 0]], longitudes[vertices[:, 1]]], axis=1)
    return polygon_haversine_diameter(vertices_coords)

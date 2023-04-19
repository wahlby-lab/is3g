import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from typing import List, Tuple 
from .linalg import spatial_binning_matrix, connectivity_matrix
import random

def knn_undirected_edges(xy:np.ndarray, k:int)->List[Tuple[int,int]]:
    """
    Computes the K-Nearest Neighbors (KNN) graph for a set of points in two dimensions.

    Args:
    - xy (np.ndarray): A two-dimensional numpy array of shape (N, 2) representing the N points.
    - k (int): The number of nearest neighbors to consider.

    Returns:
    - A list of tuples representing the edges of the KNN graph. Each tuple contains two integers
      representing the indices of the two points that form an edge.

    Example:
    ```
    >>> xy = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> knn_undirected_edges(xy, 2)
    [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
    ```
    """
    _, lil = cKDTree(xy).query(xy, k+1)
    return list({(min(i,j),max(i,j)) for i,inds in enumerate(lil) for j in inds if i != j})
    
def radius_undirected_edges(xy:np.ndarray, r:float)->List[Tuple[int,int]]:
    """
    Given a set of 2D points `xy`, compute a list of undirected edges where
    each edge is represented by a tuple `(i, j)` that connects the points with
    indices `i` and `j`. An edge is included in the output list if the
    Euclidean distance between its endpoints is less than or equal to `r`.

    Args:
        xy (np.ndarray): A 2D NumPy array with shape `(n, 2)` containing the
            `x` and `y` coordinates of `n` points.
        r (float): A non-negative scalar representing the maximum Euclidean
            distance between two connected points.

    Returns:
        A list of undirected edges, represented by tuples `(i, j)` where `i`
        and `j` are integers between 0 and `n-1`. Each tuple connects the
        points with indices `i` and `j` and the distance between these points
        is less than or equal to `r`. No duplicates are returned, i.e., the
        output list does not contain edges that connect the same pair of points
        in reverse order.
    """
    lil = cKDTree(xy).query_ball_point(xy, r)
    return list({(min(i,j),max(i,j)) for i,inds in enumerate(lil) for j in inds if i != j})
    
def _choose_one(data):
    if len(data):
        return data[random.randint(0,len(data)-1)]
    
def _choose_k(data, k):
    if len(data):
        return [data[random.randint(0,len(data)-1)] for _ in range(k)]
    return []



def distant_undirected_edges(x:np.ndarray, k:int, r_min:float, r_max:float, bin_width:float=None):
    """
    Compute a list of distant undirected edges based on a set of points in 2D or 3D space.

    Parameters
    ----------
    x : array_like, shape (npoints, ndim)
        The coordinates of the points in 2D or 3D space.
    k : int
        The number of edges to create for each point.
    r_min : float
        The minimum distance between two points for an edge to be created.
    r_max : float
        The maximum distance between two points for an edge to be considered.
    bin_width : float or None, optional
        The bin width to use for spatial binning. If None, a default value of r_min / 3.0 is used.

    Returns
    -------
    edges : list of tuples
        A list of undirected edges as tuples of point indices.

    """
        
    if bin_width is None:
        bin_width = r_min / 3.0

    # Bin points
    bin_matrix = spatial_binning_matrix(x, bin_width)
    # Keep only non-empty bins
    non_empty_bins = bin_matrix.sum(axis=1).A.flatten() > 0
    # Keep only id of non_empty bins
    bin_matrix = bin_matrix[non_empty_bins]
    bin_ids = bin_matrix.argmax(axis=0).A.flatten()
    # Get number of points
    npoints = len(x)
    # Get location of the bins
    bin_locations = (bin_matrix @ x) / bin_matrix.sum(axis=1)
    bin_locations = bin_locations.A

    # Get neighboring bins
    neighboring_bins = connectivity_matrix(
        bin_locations/bin_width, 
        method='radius', 
        r=r_max/bin_width, 
        include_self=False
    )
    
    # Get size of each bins
    bin_matrix = bin_matrix.tolil()
    neighboring_bins = neighboring_bins.tolil()

    # Neighboring bins lut
    neighboring_bins_lut = neighboring_bins.rows.copy()
    neighboring_bins_lut = {i : [j for j in n if np.linalg.norm(bin_locations[i]-bin_locations[j]) > r_min] for i,n in enumerate(neighboring_bins_lut)}

    # Compute edges
    edges = [
        (i, _choose_one(bin_matrix.rows[bin_id])) for i in range(npoints) for bin_id in _choose_k(neighboring_bins_lut[bin_ids[i]],  k) if len(neighboring_bins_lut[bin_ids[i]])
    ]

    # Make undirected
    edges = list(set([(min(e0,e1),max(e0,e1)) for (e0,e1) in edges]))
    return edges


class PairSampler:

    def __init__(self, xy:np.ndarray, neighbor_max_distance:float, non_neighbor_distance_interval:Tuple[float,float]):
        """
        Initializes the PairSampler object.

        Args:
            xy (np.ndarray): Array of shape (n, 2) containing the (x, y) coordinates of the points.
            neighbor_max_distance (float): Maximum distance between two points for them to be considered neighbors.
            non_neighbor_distance_interval (Tuple[float, float]): Tuple containing the minimum and maximum distance between two
            points for them to be considered non-neighbors.

        Returns:
            None
        """

        self.r_min = non_neighbor_distance_interval[0]
        # Find positive neighbors
        self._positive_neighbors = cKDTree(xy).query_ball_point(xy, neighbor_max_distance)
        # No self loops
        self._positive_neighbors = [[j for j in n if i != j] for i,n in enumerate(self._positive_neighbors)]
        self.xy = xy
        # Bin data
        bin_width = non_neighbor_distance_interval[0] / 3.0
        bin_matrix = spatial_binning_matrix(xy, bin_width)

        # Keep only non-empty bins
        non_empty_bins = bin_matrix.sum(axis=1).A.flatten() > 0
        # Keep only id of non_empty bins
        bin_matrix = bin_matrix[non_empty_bins]
        self._bin_ids = bin_matrix.argmax(axis=0).A.flatten()
        # Get number of points
        self._points = list(np.arange(len(xy)))
        # Get location of the bins
        bin_locations = (bin_matrix @ xy) / bin_matrix.sum(axis=1)
        bin_locations = bin_locations.A

        self._neighboring_bins = cKDTree(bin_locations/bin_width).query_ball_point(bin_locations/bin_width, non_neighbor_distance_interval[1]/bin_width)
        p = np.array([i for i,n in enumerate(self._neighboring_bins) for _ in n])
        q = np.array([j for i,n in enumerate(self._neighboring_bins) for j in n])
        dist = np.linalg.norm(bin_locations[p]-bin_locations[q],axis=1)
        adj = sp.csr_matrix((dist,(p,q)), shape=(len(bin_locations), len(bin_locations)))
        adj = adj > non_neighbor_distance_interval[0]
        self._neighboring_bins = adj.tolil().rows
        self._bin_matrix = bin_matrix.tolil().rows

        # Prepare a queue
        self._queue = [i for i in range(len(self._points))]
        self._counter = 0
        random.shuffle(self._queue)

    def sample(self, neighbor:bool) -> Tuple[int,int]:
        """
        Sample a pair of points from the provided data array.

        Args:
            neighbor (bool): A boolean indicating whether to sample a pair of
                neighboring points or not.

        Returns:
            A tuple of two integers representing the indices of the sampled
            points. If `neighbor=True`, the sampled pair of points will be
            neighbors, i.e., their distance will be less than
            `neighbor_max_distance`. If `neighbor=False`, the sampled pair of
            points will not be neighbors, i.e., their distance will be greater
            than or equal to `r_min`.
        """

        if neighbor:
            while True:
                p1 = self._pick_point()
                if len(self._positive_neighbors[p1]):
                    p2 = _choose_one(self._positive_neighbors[p1])
                    return p1,p2
        else:
            while True:
                p1 = self._pick_point()
                bin_id_p1 = self._bin_ids[p1]
                if len(self._neighboring_bins[bin_id_p1]):
                    bin_id_p2 = _choose_one(self._neighboring_bins[bin_id_p1])
                    if len(self._bin_matrix[bin_id_p2]):
                        p2 = _choose_one(self._bin_matrix[bin_id_p2])
                        return p1, p2

    def _pick_point(self):
        if self._counter == len(self._queue):
            self._queue = [i for i in range(len(self._points))]
            random.shuffle(self._queue)
            self._counter = 0
        output = self._queue[self._counter]
        self._counter += 1
        return output

    
    
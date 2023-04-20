import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.preprocessing import OneHotEncoder
from typing import Optional


def connectivity_matrix(
    xy: np.ndarray,
    method="knn",
    k: int = 5,
    r: Optional[float] = None,
    include_self: bool = False,
) -> sp.spmatrix:
    """
    Compute the connectivity matrix of a dataset based on either k-NN or radius search.

    Parameters
    ----------
    xy : np.ndarray
        The input dataset, where each row is a sample point.
    method : str, optional (default='knn')
        The method to use for computing the connectivity.
        Can be either 'knn' for k-nearest-neighbors or 'radius' for radius search.
    k : int, optional (default=5)
        The number of nearest neighbors to use when method='knn'.
    r : float, optional (default=None)
        The radius to use when method='radius'.
    include_self : bool, optional (default=False)
        If the matrix should contain self connectivities.

    Returns
    -------
    A : sp.spmatrix
        The connectivity matrix, with ones in the positions where two points are connected.
    """
    if method == "knn":
        A = kneighbors_graph(xy, k, include_self=include_self)
    else:
        A = radius_neighbors_graph(xy, r, include_self=include_self)
    return A


def attribute_matrix(
    cat: np.ndarray, unique_cat: np.ndarray = "auto", return_encoder: bool = False
):
    """
    Compute the attribute matrix from categorical data, based on one-hot encoding.

    Parameters
    ----------
    cat : np.ndarray
        The categorical data, where each row is a sample and each column is a feature.
    unique_cat : np.ndarray
        Unique categorical data used to setup up the encoder. If "auto", unique categories
        are automatically determined from cat.
    return_encoder : bool, optional (default=False)
        Whether to return the encoder object, in addition to the attribute matrix and categories list.

    Returns
    -------
    y : sp.spmatrix
        The attribute matrix, in sparse one-hot encoding format.
    categories : list
        The categories present in the data, as determined by the encoder.
    encoder : OneHotEncoder
        The encoder object, only returned if `return_encoder` is True.
    """
    X = np.array(cat).reshape((-1, 1))
    if not isinstance(unique_cat, str):
        unique_cat = [np.array(unique_cat)]
    else:
        if unique_cat != "auto":
            raise ValueError("`unique_cat` must be a numpy array or the string `auto`.")
    encoder = OneHotEncoder(categories=unique_cat, sparse=True, handle_unknown="ignore")
    encoder.fit(X)
    y = encoder.transform(X)
    categories = list(encoder.categories_[0])
    if return_encoder:
        return y, categories, encoder
    return y, categories


def degree_matrix(A: sp.spmatrix) -> sp.spmatrix:
    """
    Calculates the degree matrix of a given matrix.

    Parameters:
    -----------
    A : sp.spmatrix
        The input matrix.

    Returns:
    --------
    sp.spmatrix
        The degree matrix of the input matrix.
    """
    D = np.array(A.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(D, 0))


def _adj2laplacian(A: sp.spmatrix, return_degree: bool = False) -> sp.spmatrix:
    """
    Converts a sparse matrix representation of an affinity graph to a Laplacian matrix.

    Parameters:
    -----------
    A : sp.spmatrix
        The input affinity matrix.
    return_degree : bool, optional
        If True, returns both the Laplacian matrix and the degree matrix. Default is False.

    Returns:
    --------
    sp.spmatrix or tuple
        If `return_degree` is False, returns the Laplacian matrix.
        If `return_degree` is True, returns a tuple containing the Laplacian matrix and the degree matrix.
    """
    aff_tilde = A + sp.eye(*A.shape)
    D = degree_matrix(aff_tilde)
    L = D - aff_tilde
    return (L, D) if return_degree else L


def proximity_matrix(A: sp.spmatrix, gamma: float = 1.0, hops: int = 1) -> sp.spmatrix:
    """
    Calculates the proximity matrix of a given matrix.

    Parameters:
    -----------
    A : sp.spmatrix
        The input matrix.
    gamma : float, optional
        The decay factor for the proximity matrix. Default is 1.0.
    hops : int, optional
        The number of hops to calculate. Default is 1.

    Returns:
    --------
    sp.spmatrix
        The proximity matrix of the input matrix.
    """
    L, D = _adj2laplacian(A, return_degree=True)
    _I = sp.eye(*D.shape)
    D_inv = sp.csr_matrix(sp.diags(1.0 / (D.diagonal() + 1e-12), 0))
    P = (_I - gamma * D_inv @ L) ** hops
    return P


def proximity_matrix_multiply(
    A: sp.spmatrix, y: sp.spmatrix, gamma: float = 1.0, hops: int = 1
) -> sp.spmatrix:
    """
    Calculates the proximity matrix of a given matrix.

    Parameters:
    -----------
    A : sp.spmatrix
        The input matrix.
    y : sp.spmatrix
        The matrix that is to be multiplied by the proximity matrix
    gamma : float, optional
        The decay factor for the proximity matrix. Default is 1.0.
    hops : int, optional
        The number of hops to calculate. Default is 1.

    Returns:
    --------
    sp.spmatrix
        The multiplication of the proximity matrix and y.
    """
    P = proximity_matrix(A, gamma, hops=1)
    out = P @ y
    if hops > 1:
        for _ in range(hops - 1):
            out = P @ out
    return out


def maximal_degree_matrix(A: sp.spmatrix) -> sp.spmatrix:
    """
    Given an adjacency matrix A, returns a binary matrix representing a maximal independent set of the graph described by A.
    A maximal independent set is a set of vertices such that no two vertices are connected by an edge, and it is not possible
    to add any vertices to the set.

    Parameters:
    A (sp.spmatrix): The input adjacency matrix.

    Returns:
    sp.spmatrix: A binary matrix representing a maximal independent set of the graph described by A.

    """
    # Sort vertices based on degree
    degree = A.sum(axis=1).A.ravel()
    vertices_sorted = np.flip(np.argsort(degree))
    independent, dependent = set({}), set({})
    neighbors = A.tolil(copy=True).rows
    for i in vertices_sorted:
        neighbor = neighbors[i]
        if i not in dependent:
            independent.add(i)
            dependent.update(neighbor)
        dependent.add(i)
    # Format output as a sparse matrix
    n = len(independent)
    values = np.ones(n)
    cols = np.sort(list(independent))
    rows = np.arange(n)
    return sp.csr_matrix((values, (rows, cols)), shape=(n, A.shape[0]))


def spatial_binning_matrix(
    xy: np.ndarray, box_width: float, return_size: bool = False
) -> sp.spmatrix:
    """
    Compute a sparse matrix that indicates which points in a point cloud fall in which hyper-rectangular bins.

    Parameters:
    points (numpy.ndarray): An array of shape (N, D) containing the D-dimensional coordinates of N points in the point cloud.
    box_width (float): The width of the bins in which to group the points.
    return_size (bool): Wether the sie of the grid should be returned. Default False.

    Returns:
    sp.spmatrix: A sparse matrix of shape (num_bins, N) where num_bins is the number of bins. The matrix is such that the entry (i,j) is 1 if the j-th point falls in the i-th bin, and 0 otherwise.

    Example:
    >>> points = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2, 2, 2]])
    >>> bin_matrix = spatial_binning_matrix(points, 1)
    >>> print(bin_matrix.toarray())
    [[1 1 0 0]
     [1 1 0 0]
     [0 0 1 1]]
    """

    # Compute shifted coordinates
    mi, ma = xy.min(axis=0, keepdims=True), xy.max(axis=0, keepdims=True)
    xys = xy - mi

    # Compute grid size
    grid = ma - mi
    grid = grid.flatten()

    # Compute bin index
    bin_ids = xys // box_width
    bin_ids = bin_ids.astype("int")
    bin_ids = tuple(x for x in bin_ids.T)

    # Compute grid size in indices
    size = grid // box_width + 1
    size = tuple(x for x in size.astype("int"))

    # All gird coordinates
    all_grid_coords = tuple(
        s.flatten() for s in np.meshgrid(*tuple(np.arange(s) for s in size))
    )
    all_linear_coords = np.ravel_multi_index(all_grid_coords, size, order="F")

    # Convert bin_ids to integers
    linear_ind = np.ravel_multi_index(bin_ids, size)

    # Create a matrix indicating which markers fall in what bin
    bin_matrix, _ = attribute_matrix(linear_ind, unique_cat=all_linear_coords)
    bin_matrix = bin_matrix.T

    return (bin_matrix, size) if return_size else bin_matrix


def kde(xy: np.ndarray, sigma: float, box_width: float = 1) -> np.ndarray:
    """
    Computes a 2D kernel density estimate for the input array of points.

    Parameters
    ----------
    xy : np.ndarray, shape (n, 2)
        An array of n 2D points for which the density estimate is computed.
    sigma : float
        The standard deviation of the Gaussian kernel used for smoothing the density.
    box_width : float, optional
        The width of each bin in the spatial binning step. Default is 1.

    Returns
    -------
    densities : np.ndarray, shape (n,)
        An array of n density estimates, one for each input point in xy.

    Notes
    -----
    The function works by first dividing the 2D space into a grid of square bins,
    each with side length `box_width`. The number of input points falling into each bin
    is counted, and the counts are used to create an image of the density function.
    The image is then smoothed using a Gaussian kernel with standard deviation `sigma`.
    Finally, the density estimate for each input point is obtained by looking up the value
    of the density image at the bin containing that point.
    """
    from scipy.ndimage import gaussian_filter

    # Find assigment matrix B, which points belong to what bin
    B, size = spatial_binning_matrix(xy, box_width=box_width, return_size=True)

    image = B.sum(axis=1).A.flatten().reshape(size)

    # Blur with a Gaussian
    image = gaussian_filter(image, sigma=sigma)

    # Find which molecule belong to what bin
    molecule2bin = (xy - xy.min(axis=0, keepdims=True)) // box_width
    molecule2bin = molecule2bin.astype("int")
    molecule2bin = tuple(x for x in molecule2bin.T)

    # Look up densities
    densities = image[molecule2bin]
    return densities


def _make_gaussian(sigma, shape):
    mu = np.arange(shape).reshape((-1, 1))
    x = np.arange(shape)
    d2 = (mu - x) ** 2
    gaussian = np.exp(-d2 / (2 * sigma * sigma)) * (d2 < (9 * sigma * sigma))
    gaussian = gaussian / gaussian.max(axis=1, keepdims=True)
    return gaussian


def kde_per_label(xy: np.ndarray, labels: np.ndarray, sigma: float):
    """
    Computes the kernel density estimation (KDE) for each label in `labels`, using the data points in `xy` as inputs.
    Returns the KDE values as an attribute matrix, and the unique labels found in `labels`.

    Parameters:
    -----------
    xy : numpy.ndarray
        A 2D numpy array of shape (n, 2) containing the x-y coordinates of the data points.
    labels : numpy.ndarray
        A 1D numpy array of length n containing the label for each data point.
    sigma : float
        The standard deviation of the Gaussian kernel to use in the KDE.

    Returns:
    --------
    Tuple of two numpy.ndarray:
        - `att`: A 2D numpy array of shape (n_labels, n_features), where n_labels is the number of unique labels in `labels`
                  and n_features is the number of attributes (columns) in `labels`. Each row represents the KDE values
                  for a single label.
        - `unique_labels`: A 1D numpy array containing the unique labels found in `labels`.
    """
    att, unqiue_labels = attribute_matrix(labels)
    adj = connectivity_matrix(xy, method="radius", r=3.0 * sigma)
    row, col = adj.nonzero()
    d2 = np.linalg.norm(xy[row] - xy[col], axis=1) ** 2
    a2 = np.exp(-d2 / (2 * sigma * sigma))
    aff = sp.csr_matrix((a2, (row, col)), shape=adj.shape)
    return aff @ att, unqiue_labels


def __kde_per_label(
    xy: np.ndarray,
    labels: np.ndarray,
    sigma: float,
    box_width: float = 1.0,
    unique_labels="auto",
    progress: bool = False,
):
    """
    OLD

    Computes a kernel density estimate (KDE) for each gene in a given attribute matrix based on the spatial locations of the
    molecules in the xy array. Uses a Gaussian filter with standard deviation sigma and a spatial binning box width of box_width.

    Args:
        xy (np.ndarray): An N x 2 array representing the spatial locations of N molecules.
        labels (np.ndarray): Categorical labels of each molecule
        sigma (float): The standard deviation of the Gaussian filter used for smoothing.
        box_width (float, optional): The width of the spatial bins used for binning the molecules. Defaults to 1.0.
        progress (bool, optional): Whether to display a progress bar during computation. Defaults to False.

    Returns:
        sp.spmatrix: An M x S sparse matrix representing the KDE for each gene, where S is the total number of spatial bins.
        unique_labels: List of labels indicating the label of each feature dimension.
    """

    A, unique_labels = attribute_matrix(labels, unique_cat=unique_labels)

    sigma = sigma / box_width
    # Number of molecules
    len(xy)

    # Compute molecule to bin matrix B.
    B, size = spatial_binning_matrix(xy, box_width=box_width, return_size=True)

    # Count frequency of molecules in each bin.
    F = (B @ A).tocsc()

    # Find number of unique molecules
    n_unique_molecules = F.shape[1]

    # Create Gaussian filters
    GX = sp.csr_matrix(_make_gaussian(sigma, size[0]))
    GY = sp.csr_matrix(_make_gaussian(sigma, size[1]).T)

    loop = range(n_unique_molecules)
    if progress:
        from tqdm.auto import tqdm

        loop = tqdm(loop)

    mask = sp.coo_matrix(B.sum(axis=1).reshape(size) > 0)

    F = [
        (GX @ F[:, i].reshape(size) @ GY).multiply(mask).reshape((-1, 1)) for i in loop
    ]

    F = sp.hstack(F).tocsr()

    # Find which molecule belong to what bin
    molecule2bin = (xy - xy.min(axis=0, keepdims=True)) // box_width
    molecule2bin = molecule2bin.astype("int")
    molecule2bin = np.ravel_multi_index(tuple(x for x in molecule2bin.T), size)

    return F[molecule2bin].tocsr(), unique_labels


def local_maxima(A: sp.spmatrix, attributes: np.ndarray):
    # Convert to list of list
    L = A + sp.eye(A.shape[0])
    L = L.tolil()

    largest_neighbor = [np.max(attributes[n]) for n in L.rows]
    maximas = set({})
    neighbors = [set(n) for n in L.rows]

    # Loop over each node
    visited = set({})
    for node in np.flip(np.argsort(attributes)):
        if node not in visited:
            if attributes[node] >= largest_neighbor[node]:
                maximas.add(node)
                visited.update(neighbors[node])
            visited.add(node)

    maximas = np.array(list(maximas))
    maximas_values = attributes[maximas]
    maximas = maximas[np.flip(np.argsort(maximas_values))]
    return maximas


def distance_filter(xy):
    from scipy.spatial import cKDTree as KDTree
    from sklearn.mixture import BayesianGaussianMixture

    # Find distance to 10th nearest neighbor
    distances, _ = KDTree(xy).query(xy, k=11)
    distances = distances[:, -1].reshape((-1, 1))
    mdl = BayesianGaussianMixture(n_components=2)
    mdl.fit(distances)
    means = mdl.means_
    fg_label = np.argmin(means.flatten())
    labels = mdl.predict(distances)
    fg_index = np.where(labels == fg_label)[0]
    return fg_index

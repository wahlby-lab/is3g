import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from .linalg import spatial_binning_matrix, attribute_matrix, connectivity_matrix
from skimage.transform import rescale
from typing import Tuple

def _find_minimum_spanning_trees(label_mask):
    labels = np.unique(label_mask)
    paths = {}
    trees = {}
    from tqdm.auto import tqdm
    for label in tqdm(labels):
        if label > 0:
            # Create a binary mask for the current label
            rc = np.where(label_mask==label)
            # Convert to array
            rc = np.vstack(rc).T
            # Convert the binary mask to a graph adjacency matrix
            edges = connectivity_matrix(rc, method='radius', r=np.sqrt(2))
            # Convert the adjacency matrix to a sparse matrix
            # Find the minimum spanning tree using Kruskal's algorithm
            mst = minimum_spanning_tree(edges)
            # Find the longest path in the minimum spanning tree
            # using Dijkstra's algoritm
            longest_path = _extract_longest_path(mst)
            longest_path = [rc[l,:] for l in longest_path]
            longest_path.append(longest_path[0])

            rr = np.vstack(mst.nonzero()).T
            mst = [(rc[r[0],:], rc[r[1],:]) for r in rr]
            

            paths[label] = np.array(longest_path)
            trees[label] = mst
    return paths, trees

def _extract_longest_path(mst):
    import warnings

    # Find the shortest paths from a starting node to all other nodes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist_matrix, predecessors = dijkstra(-mst, directed=False, return_predecessors=True)

    # Find node to start with
    start_node = dist_matrix.min(axis=1).argmin()

    # Find the node with the maximum distance (longest path)
    end_node = np.argmin(dist_matrix[start_node])
    longest_path = [end_node]

    # Trace back the longest path
    while end_node != start_node:
        end_node = predecessors[start_node,end_node]
        longest_path.append(end_node)

    # Reverse the longest path to obtain the correct order
    longest_path = longest_path[::-1]

    return longest_path


def _finite_difference_matrix(w):
    return np.eye(w,k=-1) - np.eye(w,k=1)

def _gaussian_blur_matrix(w, sigma):
    x = np.arange(w).reshape((1,-1))
    d = abs(x-x.T)
    mask = d < 3*sigma
    g = np.exp(-0.5*d**2/sigma**2) * mask
    g = g / g.max(axis=1,keepdims=True)
    return sp.csr_matrix(g)

def _compute_cell_label_mask(xy, A, gridstep, threshold, dapi_shape):
    # Compute binning matrix
    B_non_empty, grid_props = spatial_binning_matrix(xy, gridstep, return_grid_props=True, xy_min=(0,0), xy_max=[dapi_shape[1], dapi_shape[0]])

    # Get dimension of binning matrix
    _, n_cells = B_non_empty.shape

    # Get shape of full binning matrix
    grid_shape = grid_props['grid_size']
    n_pixels = np.prod(grid_shape)

    # Get coordinate of each bin
    grid_coords = grid_props['grid_coords']
    grid_coords_linear = np.ravel_multi_index(grid_coords, grid_shape, order='C')

    # Create binning matrix (which includes empty bins)
    rows, cols = B_non_empty.nonzero()
    rows = grid_coords_linear[rows]
    val = np.ones(len(cols))

    # Create a complete binning matrix (num pixels x num cells)
    B = sp.csr_matrix((val, (rows,cols)), shape=(n_pixels, n_cells))

    # Bin points on a grid
    BA = B @ A

    # Convolutions in x and y
    Gy, Gx = tuple(_gaussian_blur_matrix(w, 2.0) for w in grid_shape)

    # Compute KDE
    s = BA.shape
    h, w = grid_shape
    c = BA.shape[1]

    # Convolve in y
    kde = (Gy @ BA.reshape((h, c*w))).reshape(s).tocsr()
    kde = (kde.T.reshape((h*c,w))@ Gx.T).reshape((c,h*w)).T.tocsr()

    # Remove background
    kde = kde.multiply(kde > threshold)

    # Find label mask
    label_mask = kde.tocoo().argmax(axis=1).A.flatten().reshape(grid_shape)

    return label_mask, grid_props




def _compute_boundary_label_mask(im):
    Dy, Dx = tuple(_finite_difference_matrix(w) for w in im.shape)
    not_bg = im > 0
    y = Dy @ im
    x = im @ Dx.T
    return im * (((y != 0) | (x != 0)) & not_bg)

def _create_raster(trees, shape):
    raster = np.zeros(shape, dtype='bool')
    ind = np.vstack(list(trees.values()))
    raster[ind[:,0],ind[:,1]] = True
    return raster


def create_binary_edges(xy: np.ndarray, cell: np.ndarray, gridstep: float, threshold:float, dapi_shape: Tuple[int,int]) -> np.ndarray:
    A, _ = attribute_matrix(cell)
    print('Binning ...')
    cell_label_mask, grid_props = _compute_cell_label_mask(xy, A, gridstep, threshold, dapi_shape)
    print('Compute label mask ...')
    boundary_label_mask = _compute_boundary_label_mask(cell_label_mask)
    print('Finding MST ...')
    trees, _ = _find_minimum_spanning_trees(boundary_label_mask)
    print('Creating rasters ...')
    raster = _create_raster(trees, grid_props['grid_size'])
    print('Rescale ...')
    raster = rescale(raster, gridstep).T
    return raster
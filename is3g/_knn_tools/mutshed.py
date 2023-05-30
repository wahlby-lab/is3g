from math import isnan
from typing import Any, Dict, List, Set, Tuple


class UnionFind:
    """
    Union-find data structure. Based on Josiah Carlson's code,
    https://code.activestate.com/recipes/215912/
    with additional changes by D. Eppstein.
    http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py
    """

    def __init__(self):
        self.node_weights = {}
        self.node_parent = {}

    def __getitem__(self, object):
        if object not in self.node_parent:
            self.node_parent[object] = object
            self.node_weights[object] = 1
            return object

        path = [object]
        root = self.node_parent[object]
        while root != path[-1]:
            path.append(root)
            root = self.node_parent[root]

        for ancestor in path:
            self.node_parent[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.node_parent)

    def union(self, *objects):
        roots = iter(
            sorted(
                {self[x] for x in objects},
                key=lambda r: self.node_weights[r],
                reverse=True,
            )
        )
        try:
            root = next(roots)
        except StopIteration:
            return

        for r in roots:
            self.node_weights[root] += self.node_weights[r]
            self.node_parent[r] = root

        path = [root]
        root = self.node_parent[root]

        for ancestor in path:
            self.node_parent[ancestor] = root


def _is_mutex(
    roots: Tuple[int, int], inclusive_set: UnionFind, exclusive_set: Dict[int, Set[int]]
):
    return (roots[1] in exclusive_set[roots[0]]) or (
        roots[0] in exclusive_set[roots[1]]
    )


def _is_connected(roots: Tuple[int, int]):
    return roots[0] == roots[1]


def _replace(s, remove, add):
    s.remove(remove)
    s.add(add)


def _inheret(roots: Tuple[int, int], incl: UnionFind, excl: Dict[int, Set[int]]):
    heaviest, lightest = tuple(
        sorted({r for r in roots}, key=lambda r: incl.node_weights[r], reverse=True)
    )
    excl[heaviest] = excl[heaviest].union(excl[lightest])
    # Replace old reference to the lightest set
    # with reference to the heavier set.
    # This will remove redundant mutex constraints.
    for v in excl[lightest]:
        _replace(excl[v], lightest, heaviest)
    excl[lightest] = set({})


def _merge(
    ij: Tuple[int, int],
    roots: Tuple[int, int],
    inclusive_set: UnionFind,
    exclusive_set: Dict[int, Set[int]],
):
    _inheret(roots, inclusive_set, exclusive_set)
    inclusive_set.union(ij[0], ij[1])


def _add_mutex(roots: Tuple[int, int], exclusive_set: Dict[int, Set[int]]):
    exclusive_set[roots[0]].add(roots[1])
    exclusive_set[roots[1]].add(roots[0])


def _make_output(vertices: List[int], inclusive_set: UnionFind):
    labels = [inclusive_set[v] for v in vertices]
    unique_labels = list({label for label in labels})

    # Remap labels to start from 0
    label2newlabel = {unique_label: i for i, unique_label in enumerate(unique_labels)}
    return {v: label2newlabel[label] for v, label in zip(vertices, labels)}


def _make_edge_iterator(edge_weights: Dict[Tuple[int, int], float], progress):
    edges = list(edge_weights.keys())
    edges = sorted(edges, key=lambda t: abs(edge_weights[t]), reverse=True)

    if progress:
        from tqdm.auto import tqdm

        edge_iter: Any = tqdm(edges, total=(len(edges)))
    else:
        edge_iter = edges
    for edge in edge_iter:
        w = edge_weights[edge]
        yield edge, w


def _check_edges(edge_weights: Dict[Tuple[int, int], float]):
    for edge, weight in edge_weights.items():
        if edge[1] <= edge[0]:
            raise ValueError(
                "Edges must be tuples where the first entry is smaller than the second."
            )
        if isnan(weight):
            raise ValueError("Edge weight must not be nan.")


def mutshed(
    edge_weights: Dict[Tuple[int, int], float], progress: bool = False
) -> Dict[int, int]:
    """
    Partitions a signed graph using the Mutex Watershed algorithm
    (https://arxiv.org/pdf/1904.12654.pdf).

    Args:
        edge_weights (Dict[Tuple[int,int],float]): A dictionary containing the
            signed weights of the edges in the graph. Each key is a tuple of two
            integers representing the vertices of the edge, and each value is a float
            representing the attractive (positive)  or repulsive (negative) strength.
            of the edge.
        progress (bool, optional): Whether to display a progress bar while the
            algorithm is running. If True, the function requires the `tqdm` package
            to be installed. Defaults to False.

    Returns:
        Dict[int, int]: A dictionary with that maps each vertex to a component
        partitioned by the algorithm.

    Raises:
        ImportError: If `progress` is True but the `tqdm` package is not installed.
    """

    _check_edges(edge_weights)
    vertices = list({vertex for edge in edge_weights.keys() for vertex in edge})
    edge_iterator = _make_edge_iterator(edge_weights, progress)
    inclusive_set = UnionFind()
    exclusive_set: Dict[int, set] = {v: set({}) for v in vertices}

    active_set = {}
    for edge, weight in edge_iterator:
        roots = (inclusive_set[edge[0]], inclusive_set[edge[1]])
        if weight > 0:
            if not _is_mutex(roots, inclusive_set, exclusive_set):
                if not _is_connected(roots):
                    _merge(edge, roots, inclusive_set, exclusive_set)
                    active_set[edge] = weight
        else:
            if not _is_connected(roots):
                _add_mutex(roots, exclusive_set)
                active_set[edge] = weight

    out = _make_output(vertices, inclusive_set)
    return out

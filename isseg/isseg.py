import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import cKDTree as KDTree
from sklearn.mixture import BayesianGaussianMixture
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ._knn_tools.edges import (
    PairSampler,
    distant_undirected_edges,
    knn_undirected_edges,
)
from ._knn_tools.linalg import kde_per_label
from ._knn_tools.mutshed import mutshed


def _chunker(seq, size=500000):
    # Utility function to split a sequence into chucks.
    # Useful for iterating over sparse matrices semi-efficiently
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def _mask_intracellular(xy: np.ndarray):
    features, _ = KDTree(xy).query(xy, k=11)
    features = features[:, -1].reshape((-1, 1))
    mdl = BayesianGaussianMixture(n_components=2)
    mdl.fit(features)
    means = mdl.means_
    bg_label = np.argmin(means.flatten())
    labels = mdl.predict(features)
    bg_mask = np.where(labels == bg_label)[0]
    return bg_mask


class PairDataset(Dataset):
    def __init__(
        self,
        features: sp.spmatrix,
        xy: np.ndarray,
        neighbor_max_distance: float,
        non_neighbor_max_distance_interval: Tuple[float, float],
    ):
        self.features = features
        self.sampler = PairSampler(
            xy, neighbor_max_distance, non_neighbor_max_distance_interval
        )
        self.xy = xy
        self.buffer: List[Any] = []
        self._counter = np.inf

    def __len__(self):
        return self.features.shape[0]

    def _fill_buffer(self):
        labels = np.random.randint(0, 2, size=10000)
        pq = np.array([self.sampler.sample(label == 1) for label in labels]).T
        t1, t2 = self.features[pq[0], :].A, self.features[pq[1], :].A
        t1 = torch.from_numpy(t1).float()
        t2 = torch.from_numpy(t2).float()
        self.buffer = [(labels[i], (t1[i], t2[i])) for i in range(len(labels))]
        self._counter = 0

    def __getitem__(self, index):
        if self._counter < len(self.buffer):
            output = self.buffer[self._counter]
            self._counter += 1
            return output
        else:
            self._fill_buffer()
            return self.__getitem__(None)


class SiameseNet(nn.Module):
    """
    A Siamese neural network for computing the similarity between pairs of data points.

    Parameters
    ----------
    input_dim : int
        The number of input features.
    hidden_dim : int, optional (default=64)
        The number of neurons in the hidden layer.
    output_dim : int, optional (default=32)
        The number of neurons in the output layer.

    Methods
    -------
    forward(x1, x2)
        Computes the output of the Siamese network for two input tensors x1 and x2.
    score_edge(x, y, device='cpu', attractive=True)
        Computes the score of an edge between two data points x and y.
    score_edge_sparse(x, y, device='cpu', attractive=True)
        Computes the score of an edge between two sparse matrices x and y.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        """
        Initializes a new instance of the SiameseNet class.

        Parameters
        ----------
        input_dim : int
            The number of input features.
        hidden_dim : int, optional (default=64)
            The number of neurons in the hidden layer.
        output_dim : int, optional (default=32)
            The number of neurons in the output layer.
        """

        super(SiameseNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.fc4 = nn.Linear(1, 1)

    def forward(self, x1: torch.tensor, x2: torch.tensor):
        """
        Computes the output of the Siamese network for two input tensors x1 and x2.

        Parameters
        ----------
        x1 : torch.tensor
            The first input tensor of shape (batch_size, input_dim).
        x2 : torch.tensor
            The second input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.tensor
            The output tensor of shape (batch_size, 1).
        """

        out1 = self.fc1(x1)
        out1 = nn.functional.elu(out1)
        out2 = self.fc1(x2)
        out2 = nn.functional.elu(out2)

        distance = nn.functional.pairwise_distance(out1, out2, p=2).view((-1, 1))
        output = self.fc4(distance)
        return output

    def score_edge(self, x: torch.tensor, y: torch.tensor, attractive: bool = True):
        """
        Computes the score of an edge between two nodes x and y based on the output of
        the Siamese network.

        Parameters:
        -----------
        x : torch.tensor of shape (batch_size, input_dim)
            The tensor representing the first node.
        y : torch.tensor of shape (batch_size, input_dim)
            The tensor representing the second node.
        device : str, optional (default='cpu')
            The device on which to run the computation.
        attractive : bool, optional (default=True)
            If True, returns the attractive force score (i.e., higher score means the
                nodes are more likely to be in the same cluster).
            If False, returns the repulsive force score (i.e., higher score means the
                nodes are less likely to be in the same cluster).

        Returns:
        --------
        score : Union[float, np.ndarray]
            The score of the edge between x and y. If batch_size=1, returns a float.
            Otherwise, returns a np.ndarray of shape (batch_size,).
        """
        z = self.forward(x, y)
        z = torch.sigmoid(z)
        attractive_force = z
        repulsive_force = 1.0 - z
        if attractive:
            score = attractive_force
        else:
            score = repulsive_force
        if score.numel() == 1:
            return score.item()
        return score.flatten().detach().cpu().numpy()

    def score_edge_sparse(
        self,
        x: sp.spmatrix,
        y: sp.spmatrix,
        device: str = "cpu",
        attractive: bool = True,
    ):
        """
        Computes the score (attractive or repulsive force) for the edge connecting the
        vectors x and y.

        Parameters:
        -----------
        x : sp.spmatrix of shape (1, n_features)
            The sparse matrix representing the first vector of the edge.
        y : sp.spmatrix of shape (1, n_features)
            The sparse matrix representing the second vector of the edge.
        device : str, optional (default='cpu')
            The device on which to perform the computations.
        attractive : bool, optional (default=True)
            Whether to compute the attractive force or the repulsive force.

        Returns:
        --------
        score : float or np.ndarray
            The computed score (attractive or repulsive force) for the edge connecting x
            and y.
        """
        x, y = torch.tensor(x.A).float().to(device), torch.tensor(y.A).float().to(
            device
        )
        return self.score_edge(x, y, attractive)


def isseg(
    data: pd.DataFrame,
    x: str,
    y: str,
    label: str,
    radius: float,
    remove_background: bool = True,
    labels_to_ignore_in_kde: Optional[Sequence] = None,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]],
]:
    # Set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    generator = torch.Generator()
    generator.manual_seed(0)

    xy = data[[x, y]].to_numpy()
    labels = data[label].to_numpy()
    cell_diameter = 2 * radius

    if remove_background:
        ind = _mask_intracellular(xy)
    else:
        ind = np.arange(len(xy))

    cluster_labels = np.ones(len(xy), dtype="int") * -1
    xy_filt, labels_filt = xy[ind], labels[ind]
    cluster = _cluster(xy_filt, labels_filt, cell_diameter, labels_to_ignore_in_kde)
    cluster_labels[ind] = cluster
    cluster_labels = cluster_labels + 1
    return cluster_labels


def _cluster(xy, labels, cell_diameter, ignore_in_kde):
    scale = cell_diameter / 4
    features, unique_labels = kde_per_label(xy, labels, scale)

    density = features.max(axis=1).A.flatten()
    density = density / density.max()
    if ignore_in_kde is not None:
        for id in ignore_in_kde:
            ind = np.where(np.array(unique_labels) == id)[0]
            features[:, ind] = 0
            print(f"Ignored {id}, {ind}")

    # Number of input features
    input_dim = features.shape[1]

    # Hyperparameters
    batch_size = 2048
    learning_rate = 1e-2
    num_epochs = 500

    dataset = PairDataset(
        features, xy, cell_diameter, (cell_diameter, cell_diameter * 5)
    )

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Define edges for partitioning
    short_edges = knn_undirected_edges(xy, k=15)
    long_edges = distant_undirected_edges(
        xy, k=15, r_min=cell_diameter, r_max=3 * cell_diameter
    )
    # Set up variables for early stopping

    best_loss = float("inf")
    patience = 25
    num_epochs_without_improvement = 0

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Create model
    model = SiameseNet(input_dim).to(device)

    # Setup loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loop over epochs
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        loss_avg = 0.0
        acc_avg = 0.0

        for i, (label_batch, feature_batch) in enumerate(dataloader):
            # get labels and features
            label_batch = label_batch.view((-1, 1)).float().to(device)
            x = feature_batch[0].to(device)
            y = feature_batch[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward the features
            embedding = model(x, y)
            pred = torch.sigmoid(embedding) > 0.5

            # Compute the loss and accuracy
            loss = criterion(embedding, label_batch)
            accuracy = torch.mean((pred == label_batch).float())

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            loss_avg += loss.item() / (len(dataloader))
            acc_avg += accuracy / (len(dataloader))

        # Update progress bar
        pbar.set_description(
            f"Epoch {epoch+1}: loss={loss_avg:.6f}, accuracy={acc_avg:.6f}"
        )

        # check if this is the best loss so far
        if loss_avg < best_loss:
            best_loss = loss_avg
            num_epochs_without_improvement = 0
            torch.save(
                model.state_dict(), "best-model.pt"
            )  # Should maybe save this in a temp folder somewhere

        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement >= patience:
                print(f"Stopping early at epoch {epoch} with loss {best_loss:.4f}")
                break

    model = SiameseNet(input_dim).to(device)
    model.load_state_dict(torch.load("best-model.pt"))
    model.eval()
    # Set up signed edges
    signed_edges = {}

    # Compute short edges. Bit hacky to make it fast
    pq = np.array(short_edges)
    short_edges_scores = [
        model.score_edge_sparse(
            features[pq_chunk[:, 0]],
            features[pq_chunk[:, 1]],
            device=device,
            attractive=True,
        )
        * np.minimum(density[pq_chunk[:, 0]], density[pq_chunk[:, 1]])
        for pq_chunk in _chunker(pq, size=100000)
    ]
    short_edges_scores = np.concatenate(short_edges_scores)

    # Compute long edges
    pq = np.array(long_edges)
    long_edges_scores = [
        -1
        * model.score_edge_sparse(
            features[pq_chunk[:, 0]],
            features[pq_chunk[:, 1]],
            device=device,
            attractive=False,
        )
        * np.minimum(density[pq_chunk[:, 0]], density[pq_chunk[:, 1]])
        - np.minimum(density[pq_chunk[:, 0]], density[pq_chunk[:, 1]])
        * 1e9
        * (
            np.linalg.norm(xy[pq_chunk[:, 0]] - xy[pq_chunk[:, 1]], axis=1)
            > 2 * cell_diameter
        )
        for pq_chunk in _chunker(pq, size=100000)
    ]
    long_edges_scores = np.concatenate(long_edges_scores)

    for i, edge in enumerate(short_edges):
        signed_edges[edge] = short_edges_scores[i]
    for i, edge in enumerate(long_edges):
        signed_edges[edge] = long_edges_scores[i]

    label_map = mutshed(signed_edges)
    clusters = [label_map[i] if i in label_map else -1 for i in range(len(labels))]

    return clusters

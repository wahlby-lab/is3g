# Standard library imports
import os
import random
from typing import Tuple, Union, Dict, Sequence

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from scipy.spatial import cKDTree as KDTree
from sklearn.mixture import BayesianGaussianMixture
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

# Local imports
from mutex_watershed.mutex_watershed import mutex_watershed
from _knn_tools.edges import distant_undirected_edges, knn_undirected_edges, PairSampler
from _knn_tools.linalg import kde_per_label

# Set seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Set the generator seed
generator = torch.Generator()
generator.manual_seed(0)


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
        self.buffer = []
        self._counter = np.inf

    def __len__(self):
        return self.features.shape[0]

    def _fill_buffer(self):
        labels = np.random.randint(0, 2, size=10000)
        pq = np.array([self.sampler.sample(l == 1) for l in labels]).T
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
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super(SiameseNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.fc4 = nn.Linear(1, 1)

    def forward(self, x1: torch.tensor, x2: torch.tensor):
        out1 = self.fc1(x1)
        out1 = nn.functional.elu(out1)
        out2 = self.fc1(x2)
        out2 = nn.functional.elu(out2)

        distance = nn.functional.pairwise_distance(out1, out2, p=2).view((-1, 1))
        output = self.fc4(distance)
        return output

    def score_edge(
        self,
        x: torch.tensor,
        y: torch.tensor,
        device: str = "cpu",
        attractive: bool = True,
    ):
        z = self.forward(x, y)
        z = nn.functional.sigmoid(z)
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
        x, y = torch.tensor(x.A).float().to(device), torch.tensor(y.A).float().to(
            device
        )
        return self.score_edge(x, y, device, attractive)


def istseg(
    xy: np.ndarray,
    labels: np.ndarray,
    cell_diameter: float,
    remove_background: bool = True,
    labels_to_ignore_in_kde: Sequence = None,
    return_edges: bool = False,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]],
]:
    """
    Performs cell segmentation of in situ transcriptomics data.

    Parameters:
    -----------
    xy : np.ndarray of shape (N, 2)
        The array of 2D coordinates of the points.
    labels : np.ndarray of shape (N,)
        The array of integer labels corresponding to each point.
    cell_diameter : float
        The expected diameter of the cells in the sample.
    remove_background : bool, optional (default=True)
        If True, only considers points within the cell borders (discards those with label 0).
    labels_to_ignore_in_kde : Sequence, optional (default=None)
        A sequence of integer labels to ignore when computing the KDE matrix.
    return_edges : bool, optional (default=False)
        If True, also returns the signed and active edge sets.

    Returns:
    --------
    cluster_labels : np.ndarray of shape (N,)
        The resulting labels of the cells after ISTeS.
    signed_edges : dict of {(int, int): float}, optional
        The signed edge set of the computed graph (only if return_edges=True).
    active_set : dict of {(int, int): float}, optional
        The active edge set of the partitioned graph (only if return_edges=True).
    """
    if not isinstance(xy, np.ndarray):
        raise TypeError("Input 'xy' must be a numpy array.")
    if not isinstance(labels, np.ndarray):
        raise TypeError("Input 'labels' must be a numpy array.")
    if not isinstance(cell_diameter, float):
        raise TypeError("Input 'cell_diameter' must be a float.")
    if not isinstance(remove_background, bool):
        raise TypeError("Input 'remove_background' must be a bool.")
    if labels_to_ignore_in_kde is not None and not isinstance(
        labels_to_ignore_in_kde, Sequence
    ):
        raise TypeError("Input 'labels_to_ignore_in_kde' must be a sequence.")
    if not isinstance(return_edges, bool):
        raise TypeError("Input 'return_edges' must be a bool.")
    if xy.shape[1] != 2:
        raise ValueError("Input 'xy' must have shape (N,2).")
    if labels.shape != (xy.shape[0],):
        raise ValueError("Input 'labels' must have shape (N,).")
    if cell_diameter <= 0:
        raise ValueError("Input 'cell_diameter' must be a positive float.")

    if remove_background:
        ind = _mask_intracellular(xy)
    else:
        ind = np.arange(len(xy))

    cluster_labels = np.ones(len(xy), dtype="int") * -1
    xy_filt, labels_filt = xy[ind], labels[ind]
    if return_edges:
        cluster, signed_edges, active_set = _cluster(
            xy_filt, labels_filt, cell_diameter, labels_to_ignore_in_kde
        )
        cluster_labels[ind] = cluster
        signed_edges = {(ind[e0], ind[e1]): w for (e0, e1), w in signed_edges.items()}
        active_set = {(ind[e0], ind[e1]): w for (e0, e1), w in active_set.items()}
        cluster_labels = cluster_labels + 1
        return cluster_labels, signed_edges, active_set

    else:
        cluster, _, _ = _cluster(
            xy_filt, labels_filt, cell_diameter, labels_to_ignore_in_kde
        )
        cluster_labels[ind] = cluster
        cluster_labels = cluster_labels + 1
        return cluster_labels


def _cluster(xy, labels, cell_diameter, ignore_in_kde):
    scale = cell_diameter / 4
    features, unique_labels = kde_per_label(xy, labels, scale)

    density = features.max(axis=1).A.flatten()
    density = density / density.max()
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

        for i, (label_batch, feature_batch) in tqdm(enumerate(dataloader)):
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

    label_map, active_set = mutex_watershed(signed_edges)
    clusters = [label_map[l] if l in label_map else -1 for l in range(len(labels))]

    return clusters, signed_edges, active_set


if __name__ == "__main__":
    experiment = "ISS"
    if experiment == "simulated":
        from _knn_tools._debug import plot_data

        # Load the data
        data = pd.read_csv(os.path.join("datasets", "simulated", "data.csv"))

        # Parse into numpy land
        xy = data[["x", "y"]].to_numpy()
        labels = data["gene"].to_numpy().astype("int")

        # Run the clustering
        clusters, signed_edges, active_set = istseg(xy, labels, cell_diameter=50.0)

        # Plot the results
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        plot_data(xy, labels=labels, edges=active_set, ax=axs[0])
        plot_data(xy, labels=clusters, ax=axs[1])

        plt.show()

    elif experiment == "christophe":
        for cell_diameter in [35, 45, 55, 65]:
            # Load the data
            data = pd.read_csv(os.path.join("datasets", "christophe", "data.csv"))

            # Parse into numpy land
            xy = data[["x_location", "y_location"]].to_numpy()
            labels = data["feature_name"].to_numpy()

            # Run the clustering
            clusters, signed_edges = istseg(xy, labels, cell_diameter=cell_diameter)

            # Add to results
            data["instances"] = clusters
            data.to_csv(
                os.path.join("datasets", "christophe", f"result_{cell_diameter}.csv")
            )

    elif experiment == "osmFISH":
        for cell_diameter in [100]:
            # Load the data
            data = pd.read_csv(
                os.path.join("datasets", "osmFISH", "osmFISH_cropped.csv")
            )

            # Parse into numpy land
            xy = data[["X", "Y"]].to_numpy()
            labels = data["Genes"].to_numpy()
            numeric_map = {u: i for i, u in enumerate(np.unique(labels))}
            numeric_labels = np.array([numeric_map[l] for l in labels])

            # Run the clustering
            clusters, signed_edges, active_set = istseg(
                xy,
                labels,
                cell_diameter=cell_diameter,
                labels_to_ignore_in_kde=["cell_center"],
            )

            # Add to results
            data["instances"] = clusters
            data.to_csv(
                os.path.join(
                    "datasets", "osmFISH", f"osmFISH_cropped_result{cell_diameter}.csv"
                )
            )
    elif experiment == "osmFISH-full":
        for cell_diameter in [75, 100, 125, 150, 175]:
            # Load the data
            data = pd.read_csv(
                os.path.join("datasets", "osmFISH", "osmFISH_all_data+methods.csv")
            )

            # Parse into numpy land
            xy = data[["X", "Y"]].to_numpy()
            labels = data["Genes"].to_numpy()
            numeric_map = {u: i for i, u in enumerate(np.unique(labels))}
            numeric_labels = np.array([numeric_map[l] for l in labels])

            # Run the clustering
            clusters, signed_edges, active_set = istseg(
                xy,
                labels,
                cell_diameter=cell_diameter,
                labels_to_ignore_in_kde=["cell_center"],
            )

            # Add to results
            data["instances"] = clusters
            data.to_csv(
                os.path.join(
                    "datasets",
                    "osmFISH",
                    "results",
                    f"osmFISH_full_result{cell_diameter}.csv",
                )
            )

    elif experiment == "ISS":
        for cell_diameter in [30, 50, 70, 90]:
            data = pd.read_csv(os.path.join("datasets", "iss", "ISS_all_cmaps.csv"))
            # Parse into numpy land
            xy = data[["X", "Y"]].to_numpy()
            labels = data["Genes"].to_numpy()
            # Run the clustering
            clusters, signed_edges, active_set = istseg(
                xy,
                labels,
                cell_diameter=cell_diameter,
                labels_to_ignore_in_kde=["cell_center"],
            )

            # Add to results
            data["instances"] = clusters
            data.to_csv(
                os.path.join(
                    "datasets", "iss", "results", f"iss_full_result{cell_diameter}.csv"
                )
            )

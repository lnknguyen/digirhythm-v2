import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import HDBSCAN


SEED = 2508


def hdbscan(umaps):
    """
    Perform HDBSCAN clustering on UMAP embeddings.

    Parameters:
      - umaps: array-like, shape (n_samples, 2)
      - eps: float, DBSCAN eps parameter (default 0.5)
      - min_samples: int, DBSCAN min_samples parameter (default 5)

    Returns:
      - labels: array of cluster labels
    """

    clustering = HDBSCAN(min_samples=10, min_cluster_size=1000)

    labels = clustering.fit_predict(umaps)
    return labels


def plot_umap_with_hdbscan(umap_cluster):
    """
    Plot UMAP embeddings colored by DBSCAN cluster labels and return the matplotlib Figure.

    Parameters:
      - umap_cluster: tuple of (components, labels)
          * components: array-like of shape (n_samples, 2)
          * labels: array-like of cluster labels (same length as components)

    Returns:
      - fig: a matplotlib Figure object containing the plot.
    """
    components, labels = umap_cluster

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        components["UMAP_1"], components["UMAP_2"], c=labels, cmap="viridis", alpha=0.7
    )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title("UMAP Projection with DBSCAN Clusters")
    fig.colorbar(scatter, ax=ax, label="Cluster Label")

    return fig


def main(input_fns, output_fns, params):

    # Load data
    uembeddings = pd.read_csv(input_fns[0], index_col=0)

    cluster_labels = hdbscan(uembeddings)

    fig = plot_umap_with_hdbscan((uembeddings, cluster_labels))

    fig.savefig(snakemake.output[0], dpi=200, bbox_inches="tight")


if __name__ == "__main__":

    main(snakemake.input, snakemake.output, snakemake.params)

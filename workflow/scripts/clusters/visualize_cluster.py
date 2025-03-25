import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def centroids_heatmap(centroids_df):

    # Get unique split
    # Identify unique splits
    splits = df["split"].unique()
    num_splits = len(splits)

    # Determine subplot layout
    n_cols = 2
    n_rows = max(1, int(np.ceil(num_splits / n_cols)))  # Ensure at least 1 row

    # Create figure for subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))

    # Convert axes to an iterable even if there is only one subplot
    if num_splits == 1:
        axes = [axes]

    # Flatten axes for easy iteration
    axes = np.ravel(axes)

    # Generate heatmaps for each split
    for i, split in enumerate(splits):
        ax = axes[i]
        subset = df[df["split"] == split].set_index("Cluster").drop(columns=["split"])

        sns.heatmap(subset, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)

        ax.set_title(f"Feature Heatmap - {split}")
        ax.set_xlabel("Features")
        ax.set_ylabel("Clusters")

    return fig


def dist_centroids_heatmap(centroids_df):

    # Compute distance centroids
    return


def main(input_fns, output_fns, params):

    # Load data
    data = pd.read_csv(input_fns[0])

    print(data)

    centroids_viz = centroids_heatmap(data)

    # Save
    centroids_viz.savefig(output_fns.centroids_viz, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

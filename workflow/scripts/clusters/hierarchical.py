import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SEED = 2508


def safe_normalize(x):
    """Safely normalize using z-score."""
    std = x.std()
    return (x - x.mean()) / (std if std != 0 else 1)


# Remove user with less observations than a given threshold
def filter_by_num_observations(df, min_obs):
    filtered_df = df.groupby("user").filter(lambda x: len(x) >= min_obs)
    return filtered_df


def create_clusters(df, features, range_n_clusters):
    """
    Creates clusters using KMeans and selects the best number of clusters
    based on Silhouette Score.
    """
    logging.info("Starting clustering process...")

    best_k = None
    best_silhouette = -1
    best_labels = None

    X = df[features]

    # Evaluate silhouette scores for different cluster numbers
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
        cluster_labels = kmeans.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        logging.info(
            f"For n_clusters = {n_clusters}, Silhouette Score: {silhouette_avg:.4f}"
        )

        if silhouette_avg > best_silhouette:
            best_k = n_clusters
            best_silhouette = silhouette_avg
            best_labels = cluster_labels

    logging.info(
        f"✅ Optimal number of clusters: {best_k} (Silhouette Score: {best_silhouette:.4f})"
    )
    return best_k, best_silhouette, best_labels


def centroid_characteristics(df, features):
    """
    Computes the centroid characteristics for each cluster.
    """
    logging.info("Calculating centroid characteristics...")
    centroid_df = df.groupby("Cluster")[features].mean().reset_index()
    logging.info("✅ Centroid characteristics calculated.")
    return centroid_df


def main(input_fns, output_fns, params):
    try:
        # Load data
        data = pd.read_csv(input_fns[0])
        features = params["features"]
        logging.info(f"✅ Data loaded with shape: {data.shape}")

        min_obs = 14
        data = filter_by_num_observations(data, min_obs=min_obs)  # 14 days
        logging.info(
            f"✅ Filter user with more than {min_obs} observations: {data.shape}"
        )

        # Normalize features per user
        df_norm = data.copy()
        df_norm[features] = df_norm.groupby(["user"])[features].transform(
            safe_normalize
        )

        logging.info("✅ Data normalized.")

        # Range of k to evaluate
        range_n_clusters = range(2, 4)

        # Create clusters
        best_k, best_silhouette, best_labels = create_clusters(
            df_norm, features, range_n_clusters
        )

        # Assign the best cluster labels to the data
        data["Cluster"] = best_labels

        # Save the clustered data
        data.to_csv(output_fns[0], index=False)
        logging.info(f"💾 Clustered data saved to: {output_fns[0]}")

        # Compute and save centroid characteristics
        centroid_df = centroid_characteristics(data, features)
        centroid_df.to_csv(output_fns[1], index=False)
        logging.info(f"💾 Centroid characteristics saved to: {output_fns[1]}")

    except Exception as e:
        logging.error(f"❌ Error during execution: {e}")


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

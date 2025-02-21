import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 2508


def main(input_fns, output_fns, params):

    # Load data
    data = pd.read_csv(input_fns[0])
    features = params["features"]
    print(f"✅ Data loaded with shape: {data.shape}")

    # Normalize features per user
    # Using z-score normalization. If the std is zero, use 1 to avoid division by zero.
    def safe_normalize(x):
        std = x.std()
        return (x - x.mean()) / (std if std != 0 else 1)

    df_norm = data.copy()

    df_norm[features] = df_norm.groupby(["user"])[features].transform(safe_normalize)
    df_norm.dropna(subset=features, inplace=True)

    # Range of k to evaluate
    range_n_clusters = range(2, 15)

    best_k = None
    best_silhouette = -1
    best_labels = None

    X = df_norm[features]

    # Find the optimal number of clusters using Silhouette Score
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
        cluster_labels = kmeans.fit_predict(X)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            f"For n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg:.4f}"
        )

        # Update best_k if this silhouette score is better
        if silhouette_avg > best_silhouette:
            best_k = n_clusters
            best_silhouette = silhouette_avg
            best_labels = cluster_labels

    # Report the best k found
    print(
        f"\n✅ Optimal number of clusters: {best_k} (Silhouette Score: {best_silhouette:.4f})"
    )

    # Assign the best cluster labels to the data
    data["Cluster"] = best_labels

    # Save the clustered data
    data.to_csv(output_fns[0], index=False)
    print(f"💾 Clustered data saved to: {output_fns[0]}")


if __name__ == "__main__":

    main(snakemake.input, snakemake.output, snakemake.params)

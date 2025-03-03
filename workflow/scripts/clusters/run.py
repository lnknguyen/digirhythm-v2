import pandas as pd
from kmeans import KMeansClustering
from gmm import GMMClustering
from sklearn.metrics import silhouette_score
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main(input_fns, output_fns, params):

    # Load data
    data = pd.read_csv(input_fns[0])
    features = params.features
    algo = snakemake.wildcards.algo

    cluster_settings = params.cluster_settings

    if algo == "kmeans":
        model = KMeansClustering(data, features, cluster_settings)
        model.init_model(n_clusters=2)
    elif algo == "gmm":
        model = GMMClustering(data, features, cluster_settings)
        model.init_model(n_components=2)
    else:
        logging.error("Algo no available", stack_info=True)
        raise ValueError("Algo not available")

    labels, centroids = model.run_pipeline()

    # Save
    if algo == "gmm":
        model.score_plot.savefig(f"out/{snakemake.wildcards.study}/gmm_score_plot.png")

    labels.to_csv(output_fns.clusters)
    centroids.to_csv(output_fns.centroids)

if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

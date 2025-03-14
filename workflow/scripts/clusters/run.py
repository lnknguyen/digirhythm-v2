import pandas as pd
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

    if algo == "gmm":
        model = GMMClustering(data, features, cluster_settings)
        model.init_model(n_components=2)
    else:
        logging.error("Algo no available", stack_info=True)
        raise ValueError("Algo not available")

    labels, centroids, model_selection_scores = model.run_pipeline()

    labels.to_csv(output_fns.clusters)
    centroids.to_csv(output_fns.centroids)
    model_selection_scores.to_csv(output_fns.scores)


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

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
    run_selection = params.run_selection
    algo = snakemake.wildcards.algo

    cluster_settings = params.cluster_settings

    if algo == "gmm":
        model = GMMClustering(data, features, cluster_settings)
        model.init_model(n_components=2)
    else:
        logging.error("Algo no available", stack_info=True)
        raise ValueError("Algo not available")

    labels, centroids, covariances, model_selection_scores = model.run_pipeline()

    if run_selection == True:
        model_selection_scores.to_csv(output_fns.scores, index=False)
    else:
        labels.to_csv(output_fns.clusters, index=False)
        centroids.to_csv(output_fns.centroids, index=False)
        covariances.to_csv(output_fns.covariances, index=False)


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

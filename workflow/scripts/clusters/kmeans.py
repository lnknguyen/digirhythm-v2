import pandas as pd
from base import BaseClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SEED = 2508


class KMeansClustering(BaseClustering):
    def init_model(self, n_clusters):
        self.model = KMeans(n_clusters=n_clusters)

    def model_selection(self, X):

        # Pre-set a k ranges
        k_range = range(2, 15)

        best_score = -1
        best_k = 2

        for k in k_range:
            self.init_model(n_clusters=k)
            labels = self.model.fit_predict(X)
            score = self.evaluate_clusters(X, labels)

            logging.info(f"k={k}, Silhouette Score={score:.4f}")

            if score > best_score:
                best_score = score

                best_k = k

        logging.info(
            f"✅ Optimal number of clusters: {best_k} with Silhouette Score: {best_score:.4f}"
        )

        # Update model
        model = KMeans(n_clusters=best_k)
        return model

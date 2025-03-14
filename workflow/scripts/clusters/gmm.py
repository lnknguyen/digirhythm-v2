import numpy as np
import pandas as pd
from base import BaseClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import det, inv

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
import logging
from scipy.spatial import distance
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SEED = 2508


class GMMClustering(BaseClustering):

    def init_model(self, n_components):
        self.model = GaussianMixture(n_components=n_components, covariance_type="full")

    def _gaussian_bhattacharyya_distance(self, mean1, cov1, mean2, cov2):
        """
        Compute the Bhattacharyya distance between two Gaussian distributions.

        Parameters:
        ----------
        mean1, mean2 : ndarray
            Mean vectors of the two Gaussian distributions (shape: [n_features]).
        cov1, cov2 : ndarray
            Covariance matrices of the two Gaussian distributions (shape: [n_features, n_features]).

        Returns:
        -------
        float
            The Bhattacharyya distance between the two distributions.
        """

        # Compute the average covariance matrix
        cov_mean = (cov1 + cov2) / 2

        # Compute the difference between the means
        mean_diff = mean1 - mean2

        # First term: Mahalanobis distance component
        term1 = 0.125 * np.dot(np.dot(mean_diff.T, inv(cov_mean)), mean_diff)

        # Second term: Covariance overlap component
        det_cov1 = det(cov1)
        det_cov2 = det(cov2)
        det_cov_mean = det(cov_mean)

        # To avoid log(0) or division by zero
        if det_cov1 <= 0 or det_cov2 <= 0 or det_cov_mean <= 0:
            return np.inf

        term2 = 0.5 * np.log(det_cov_mean / np.sqrt(det_cov1 * det_cov2))

        # Bhattacharyya distance
        distance = term1 + term2

        return distance

    def model_selection(self, X, mode="grid"):

        if mode == "manual":
            best_model, score = self.manual_model_selection(X)
        elif mode == "grid":
            best_model, score = self.grid_model_selection(X)
        else:
            raise ValueError("Mode not recognized. Valid type: 'manual', 'grid'")

        return best_model, score

    def manual_model_selection(self, X):

        logging.info("🔍 Starting Manual selection for Gaussian Mixture Model (GMM)...")

        # Define hyperparameter grid
        param_grid = {
            "n_components": range(3, 20),  # Number of clusters/components
            "covariance_type": [
                "spherical",
                "diag",
                "tied",
                "full",
            ],  # Covariance structure
        }

        # Initialize scores dictionary
        scores = {cov_type: {} for cov_type in param_grid["covariance_type"]}

        # Run 5 iterations
        for n_component in param_grid["n_components"]:
            for cov_type in param_grid["covariance_type"]:

                scores[cov_type][n_component] = {"aic": [], "bic": [], "bhatt": []}

                for i in range(5):

                    # Random seed
                    seed = random.seed(1000)

                    # Init model
                    model = GaussianMixture(
                        n_components=n_component,
                        covariance_type=cov_type,
                        init_params="k-means++",
                        random_state=seed,
                    )
                    # Fit
                    model.fit(X)

                    # Collect AIC and BIC scores
                    scores[cov_type][n_component]["aic"].append(model.aic(X))
                    scores[cov_type][n_component]["bic"].append(model.bic(X))

                    # Compute average pairwise Bhattacharyya distance for "full" covariance matrix
                    if cov_type == "full":
                        means = model.means_
                        covs = model.covariances_
                        bhatt_distances = []

                        for j in range(n_component):
                            for k in range(j + 1, n_component):
                                bhatt = self._gaussian_bhattacharyya_distance(
                                    means[j], covs[j], means[k], covs[k]
                                )
                                bhatt_distances.append(bhatt)

                        avg_bhatt = np.mean(bhatt_distances)
                        scores[cov_type][n_component]["bhatt"].append(avg_bhatt)
                    else:
                        scores[cov_type][n_component]["bhatt"].append(-1)

        # Find best model:
        # Select top 3 models with lowest BIC and pick the one with highest Bhattacharyya distance
        full_bic_scores = [
            (n, np.mean(scores["full"][n]["bic"])) for n in param_grid["n_components"]
        ]
        top_3_models = sorted(full_bic_scores, key=lambda x: x[1], reverse=True)[:3]

        best_n, _ = max(
            top_3_models, key=lambda x: np.mean(scores["full"][x[0]]["bhatt"])
        )

        # Init best model
        best_model = GaussianMixture(
            n_components=best_n,
            covariance_type="full",
            init_params="k-means++",
            random_state=SEED,
        )

        # Convert scores to dataframe

        scores = pd.concat({k: pd.DataFrame(v).T for k, v in scores.items()}, axis=0)

        scores = pd.DataFrame(scores)
        return (best_model, scores)

    def grid_model_selection(self, X, scorer="bic"):

        logging.info("🔍 Starting GridSearchCV for Gaussian Mixture Model (GMM)...")

        param_grid = {
            "n_components": range(3, 20),
            "covariance_type": ["spherical", "diag", "tied", "full"],
        }

        def bic_scorer(estimator, X):
            return -estimator.bic(X)

        def aic_scorer(estimator, X):
            return -estimator.aic(X)

        scoring_func = {"bic": bic_scorer, "aic": aic_scorer}

        if scorer not in scoring_func:
            raise ValueError("Scorer must be 'bic' or 'aic' for GridSearchCV")

        grid_search = GridSearchCV(
            GaussianMixture(random_state=SEED),
            param_grid,
            scoring=scoring_func[scorer],
            cv=3,
            verbose=1,
            n_jobs=-1,
        )
        grid_search.fit(X)

        best_model = grid_search.best_estimator_
        results_df = pd.DataFrame(grid_search.cv_results_)

        return best_model, results_df

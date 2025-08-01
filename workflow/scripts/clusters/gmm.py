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

    def _gaussian_bhattacharyya_distance(self, mu1, cov1, mu2, cov2):
        mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
        cov1, cov2 = np.atleast_2d(cov1), np.atleast_2d(cov2)

        cov_avg = 0.5 * (cov1 + cov2)
        diff = mu1 - mu2

        term1 = 0.125 * diff.T @ np.linalg.inv(cov_avg) @ diff
        term2 = 0.5 * np.log(
            np.linalg.det(cov_avg) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
        )
        return float(term1 + term2)

    def model_selection(self, X, mode="manual"):

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
            "n_components": range(5, 16),  # Number of clusters/components
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
                    # Reset the seed for each run
                    seed = random.seed(10000)

                    # Init model
                    model = GaussianMixture(
                        n_components=n_component,
                        covariance_type=cov_type,
                        init_params="k-means++",
                        random_state=seed,
                        reg_covar=1e-06,
                    )

                    # Drop na

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

        top_3_models = sorted(full_bic_scores, key=lambda x: x[1])[:3]

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
        scores_df = pd.concat(
            {cov: pd.DataFrame(n_dict).T for cov, n_dict in scores.items()},
            names=[
                "covariance_type",
                "n_components",
            ],  # give names to the two index levels
        ).reset_index()  # move both levels into columns

        scores_df = scores_df[
            ["covariance_type", "n_components", "aic", "bic", "bhatt"]
        ]
        return (best_model, scores_df)

    def grid_model_selection(self, X, scorer="bic"):

        logging.info("🔍 Starting GridSearchCV for Gaussian Mixture Model (GMM)...")

        param_grid = {
            "n_components": range(5, 21),
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
            cv=10,
            verbose=1,
            n_jobs=-1,
        )
        grid_search.fit(X)

        best_model = grid_search.best_estimator_
        results_df = pd.DataFrame(grid_search.cv_results_)

        return best_model, results_df

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

    def __post_init__(self):
        self.score_plot = None

    def init_model(self, n_components):
        self.model = GaussianMixture(n_components=2)

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

    def generate_score_plot(self, scores):
        """
        Generate a plot for AIC, BIC, and Bhattacharyya distances from GMM model selection.

        Parameters:
        ----------
        scores : dict
            Dictionary containing AIC, BIC, and Bhattacharyya distances for each
            covariance type and number of components.

        Returns:
        -------
        matplotlib.figure.Figure
            The generated plot figure.
        """

        # Prepare DataFrame for plotting
        plot_data = []

        for cov_type, components in scores.items():
            for n_component, metrics in components.items():
                # Default Bhattacharyya to NaN if not available
                bhatt_list = (
                    metrics["bhatt"]
                    if cov_type == "full" and "bhatt" in metrics
                    else [np.nan] * len(metrics["aic"])
                )

                # Iterate over AIC, BIC, and Bhattacharyya
                for aic, bic, bhatt in zip(metrics["aic"], metrics["bic"], bhatt_list):
                    plot_data.append(
                        {
                            "Covariance Type": cov_type,
                            "Number of Components": n_component,
                            "AIC": aic,
                            "BIC": bic,
                            "Bhattacharyya distance": bhatt,
                        }
                    )

        # Convert to DataFrame for easier plotting
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(plot_data)

        # Melt DataFrame for Seaborn FacetGrid
        df_melted = df.melt(
            id_vars=["Covariance Type", "Number of Components"],
            value_vars=["AIC", "BIC", "Bhattacharyya distance"],
            var_name="Score Type",
            value_name="Score",
        )

        # Compute mean and standard deviation for error bars
        df_summary = (
            df_melted.groupby(["Covariance Type", "Number of Components", "Score Type"])
            .agg(Mean_Score=("Score", "mean"), Std_Score=("Score", "std"))
            .reset_index()
        )

        # Plot using Seaborn FacetGrid with barplot
        sns.set(style="whitegrid")
        g = sns.FacetGrid(
            df_summary,
            col="Score Type",
            sharex=False,
            sharey=False,
            height=4,
            aspect=1.3,
            col_wrap=1,
        )

        g.map_dataframe(
            sns.barplot,
            x="Number of Components",
            y="Mean_Score",
            hue="Covariance Type",
            ci=None,
            capsize=0.1,
            errwidth=1,
            dodge=True,
        )

        # Add legend, titles, and labels
        g.set_titles("{col_name}")
        g.set_axis_labels("Number of Components", "Score")
        g.fig.suptitle("GMM Model Selection Metrics", y=1.05)
        g.add_legend()

        # Adjust layout
        plt.tight_layout()

        return g.fig

    def model_selection(self, X):

        logging.info("🔍 Starting GridSearchCV for Gaussian Mixture Model (GMM)...")

        # Define hyperparameter grid
        param_grid = {
            "n_components": range(5, 15),  # Number of clusters/components
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
                    scores[cov_type][n_component]["aic"].append(-model.aic(X))
                    scores[cov_type][n_component]["bic"].append(-model.bic(X))

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

        # Generate plot
        self.score_plot = self.generate_score_plot(scores)

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
        return (best_model, self.score_plot)

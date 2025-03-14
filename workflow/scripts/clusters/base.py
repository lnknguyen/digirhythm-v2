import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from abc import ABC, abstractmethod


class BaseClustering(ABC):
    def __init__(
        self,
        df: pd.DataFrame,
        features: list,
        cluster_settings: dict,
        random_state=2508,
    ):
        """
        Base class for clustering a DataFrame.

        Parameters:
        ----------
        df : pandas.DataFrame
            The input data.
        features : list
            List of feature columns for clustering.
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.df = df.copy()
        self.features = features

        self.random_state = random_state
        self.model = None

        # Score from model selection
        self.score = None

        # Cluster settings
        self.strategy = cluster_settings["strategy"]
        self.split = cluster_settings["split"]
        self.group_col = cluster_settings["group_col"]

        # Optimal cluster settings
        self.run_model_selection = cluster_settings["run_model_selection"]
        self.optimal_n_components = cluster_settings["optimal_gmm_settings"][
            "n_components"
        ]

    @abstractmethod
    def init_model(self):
        """Initialize the clustering model."""
        pass

    def split_dataset(self, df, strategy, group_col=None):
        """
        Split datasets into subsets

        strategy:
            'random' - sample without replacement 2 set of user
            'group' - Split by 'group_col' column
        """

        if strategy == "random":
            # Get unique users
            unique_users = df["user"].unique()

            # Shuffle users
            np.random.shuffle(unique_users)

            # Split users into two groups
            split_idx = len(unique_users) // 2
            users_set1 = unique_users[:split_idx]
            users_set2 = unique_users[split_idx:]

            # Create two datasets based on sampled users
            df1 = df[df["user"].isin(users_set1)]
            df2 = df[df["user"].isin(users_set2)]

            return [("split_1", df1), ("split_2", df2)]

        elif strategy == "group":

            if group_col is None:
                raise ValueError(
                    "For 'group' strategy, a 'group_col' must be specified."
                )
            if group_col not in df.columns:
                raise ValueError(f"Column '{group_col}' not found in DataFrame.")

            # Get unique values in the grouping column
            unique_groups = df[group_col].unique()

            # Create a separate DataFrame for each unique group with a tag
            split_datasets = [
                (f"group_{group}", df[df[group_col] == group])
                for group in unique_groups
            ]

            return split_datasets

        else:
            raise ValueError("Invalid strategy. Use 'random' or 'group'.")

    def filter_by_duration(self, df, threshold):

        # Count number of observations per user
        user_duration = df.groupby("user").size().reset_index(name="duration")

        filtered_users = user_duration[user_duration["duration"] >= threshold]["user"]
        filtered_df = df[df["user"].isin(filtered_users)]

        return filtered_df

    ##### Normalization ####
    def min_max_normalize_per_user(self, df):
        """
        Perform Min-Max normalization per user while safely handling features with all zeros.

        For features with the same min and max (i.e., constant or all zeros),
        assigns a normalized value of 0 to avoid division by zero.

        Returns:
        -------
        pandas.DataFrame
            DataFrame with Min-Max normalized features.
        """

        def safe_min_max(x):
            min_val = x.min()
            max_val = x.max()
            # Handle constant columns (min == max) by assigning 0
            if min_val == max_val:
                return np.zeros_like(x)
            else:
                return (x - min_val) / (max_val - min_val)

        # Apply normalization per user
        normalized_df = df.copy()
        features = self.features

        normalized_df[features] = normalized_df.groupby("user")[features].transform(
            safe_min_max
        )

        return normalized_df

    def z_score_normalize_per_user(self, df):
        """
        Perform Z-Score normalization (Z-Norm) per user while safely handling features with zero standard deviation.

        For features with zero standard deviation (i.e., constant or all zeros),
        assigns a normalized value of 0 to avoid division by zero.

        Returns:
        -------
        pandas.DataFrame
            DataFrame with Z-Score normalized features.
        """

        def safe_z_score(x):
            mean_val = x.mean()
            std_val = x.std()
            # Handle zero standard deviation by assigning 0
            if std_val == 0:
                return np.zeros_like(x)
            else:
                return (x - mean_val) / std_val

        # Apply normalization per user
        normalized_df = df.copy()
        features = (
            self.features
        )  # Assuming self.features holds the list of feature columns

        normalized_df[features] = normalized_df.groupby("user")[features].transform(
            safe_z_score
        )

        return normalized_df

    @abstractmethod
    def model_selection(self):
        """
        Return the best model and a score dataframe
        """
        pass

    def fit_predict(self, model, X):
        """
        Fit the best model and predict cluster labels.
        """
        labels = model.fit_predict(X)
        return labels

    def evaluate_clusters(self, X, labels):
        """
        Evaluate cluster quality using Silhouette Score.
        """
        score = silhouette_score(X, labels)
        print(f"Silhouette Score: {score:.4f}")
        return score

    def centroid_characteristics(self, df):
        """
        Compute centroid characteristics (mean of features per cluster).

        Returns:
        -------
        pandas.DataFrame
            DataFrame containing centroid characteristics.
        """
        if "Cluster" not in df.columns:
            raise ValueError("Clusters not assigned. Run fit_predict first.")

        centroids = df.groupby("Cluster")[self.features].mean().reset_index()
        return centroids

    def run_pipeline(self):
        """
        Run full clustering pipeline: model selection, fit, predict, and evaluate.

        Returns:
        -------
        tuple
            (cluster labels, centroids, model selection score)
        """

        # Retain users with more than 14 days of data
        self.df = self.filter_by_duration(self.df, threshold=14)  # 14 days

        # TEST
        # self.df = self.df.head(1000)

        if self.split:
            # Split dataset into subsets if self.split is True
            dfs = self.split_dataset(
                self.df, strategy=self.strategy, group_col=self.group_col
            )
        else:
            # Treat the whole dataset as a single subset
            dfs = [("full_data", self.df)]

        labels_list, centroids_list, model_selection_scores = [], [], []

        # Process each subset (either split parts or full dataset)
        for item in dfs:

            tag = item[0]
            df = item[1]

            norm_df = self.z_score_normalize_per_user(df)
            X = norm_df[self.features]

            ### Perform clustering ###

            # Select best model
            if self.run_model_selection == True:
                self.model, score = self.model_selection(X)

                # Get cluster labels
                labels = self.fit_predict(self.model, X)

                # Assign labels and split
                X["Cluster"] = labels
                X["split"] = tag
                centroids = self.centroid_characteristics(X)

                # Update model selection scores
                score["split"] = tag
                model_selection_scores.append(score)

                # Update labels
                df["Cluster"] = labels
                df["split"] = tag
                labels_list.append(df)

                # Update centrods
                centroids["split"] = tag
                centroids_list.append(centroids)

                labels_list = pd.concat(labels_list)
                centroids_list = pd.concat(centroids_list)
                model_selection_scores = pd.concat(model_selection_scores)
            else:
                self.init_model(n_components=self.optimal_n_components)

                # Get cluster labels
                labels = self.fit_predict(self.model, X)

                # Assign labels and split
                X["Cluster"] = labels
                X["split"] = tag
                centroids = self.centroid_characteristics(X)

                # Update labels
                df["Cluster"] = labels
                df["split"] = tag
                labels_list.append(df)

                # Update centrods
                centroids["split"] = tag
                centroids_list.append(centroids)

                labels_list = pd.concat(labels_list)
                centroids_list = pd.concat(centroids_list)

                # Empty scores
                model_selection_scores = pd.DataFrame()
        return labels_list, centroids_list, model_selection_scores

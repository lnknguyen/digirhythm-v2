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
        self.threshold = cluster_settings[
            "threshold"
        ]  # Number of records required to be included in the analysis

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
                (f"{group}", df[df[group_col] == group]) for group in unique_groups
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

    def get_centroids(self, df):
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

    def get_cov_matrix(self, model, *, as_array: bool = False):
        """
        Retrieve the covariance matrix (or matrices) from a fitted GaussianMixture.

        Parameters
        ----------
        model : sklearn.mixture.GaussianMixture
            A *fitted* GaussianMixture instance whose covariance matrices
            you want to inspect.
        as_array : bool, default False
            • ``False`` → return a dict mapping *component‑id → DataFrame*
              (rows / columns = ``self.features``).
            • ``True``  → return the raw NumPy stack with shape
              ``(n_components, n_features, n_features)``.

        Returns
        -------
        dict[int, pandas.DataFrame] | numpy.ndarray
            Covariance matrices per component in the format requested.

        Notes
        -----
        * Handles all four ``covariance_type`` options––``'full'``, ``'tied'``,
          ``'diag'``, ``'spherical'``––by expanding reduced forms to full
          matrices before returning them.
        * Raises ``AttributeError`` if the model has not been fitted.
        """

        if not hasattr(model, "covariances_"):
            raise AttributeError(
                "Model appears to be unfitted: no 'covariances_' attribute."
            )

        d = len(self.features)
        cov_type = model.covariance_type
        covs_raw = model.covariances_

        # ---- expand to full (n_components, d, d) ----------------------------
        if cov_type == "full":
            covs_full = covs_raw  # already full
        elif cov_type == "tied":
            covs_full = np.repeat(covs_raw[None, :, :], model.n_components, axis=0)
        elif cov_type == "diag":
            covs_full = np.array([np.diag(row) for row in covs_raw])
        elif cov_type == "spherical":
            covs_full = np.array([np.eye(d) * var for var in covs_raw])
        else:
            raise ValueError(f"Unsupported covariance_type: {cov_type}")

        # ---- return in requested format ------------------------------------
        if as_array:
            return covs_full

        return {
            k: pd.DataFrame(cov, index=self.features, columns=self.features)
            for k, cov in enumerate(covs_full)
        }

    def run_pipeline(self):
        """
        End‑to‑end clustering pipeline.

        1. Optionally split the dataset.
        2. Z‑score normalise features within person.
        3. Either select the best GMM (model‑selection) or fit a preset model.
        4. Assign cluster labels and compute centroids for every split.
        5. Return concatenated results.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            labels_df, centroids_df, model_selection_scores_df
        """

        # TEST:
        # self.df = self.df.head(1000)

        # -- 1. split dataset ---------------------------------------------------
        if self.split:
            raw_splits = self.split_dataset(
                self.df, strategy=self.strategy, group_col=self.group_col
            )

            # ---- filter each split independently ------------------------------
            splits = []
            for tag, part in raw_splits:
                part_filt = self.filter_by_duration(part, threshold=self.threshold)
                drops = len(part) - len(part_filt)
                if drops:
                    print(
                        f"[{tag}] dropped {drops} rows below {self.threshold}-day threshold"
                    )
                splits.append((tag, part_filt))

        else:
            # no splitting – just filter once
            df_filt = self.filter_by_duration(self.df, threshold=self.threshold)
            splits = [("full_data", df_filt)]

        # ---- results -----------------------------------------------------
        label_frames, centroid_frames, cov_frames, score_frames = [], [], [], []

        # ----------------------------------------------------------------------
        # 2‒4.   iterate over each split
        # ----------------------------------------------------------------------
        for tag, df_part in splits:

            #  2.  within‑person z‑score
            norm_part = self.z_score_normalize_per_user(df_part)
            X = norm_part[self.features].copy()
            # rows per user, ascending
            row_counts = df_part.groupby("user").size().sort_values()
            print(row_counts)

            #  3.  choose or initialise model
            if self.run_model_selection:

                self.model, score = self.model_selection(X)
                score_frames.append(score.assign(split=tag))
            else:
                self.init_model(n_components=self.optimal_n_components)

            #  4.  fit + predict labels
            labels = self.fit_predict(self.model, X)

            # annotate original (un‑normalised) rows
            labelled_df = df_part.assign(Cluster=labels, split=tag)
            label_frames.append(labelled_df)

            # compute centroids on the normalised features
            centroids = self.get_centroids(norm_part.assign(Cluster=labels)).assign(
                split=tag
            )
            centroid_frames.append(centroids)

            # get cov matrix
            cov_matrix = self.get_cov_matrix(self.model)
            cov_df = pd.concat(cov_matrix, names=["Cluster"])
            cov_long = (
                cov_df.stack().rename("cov").reset_index()  # or "corr"
            )  # columns: Cluster, feature_i, feature_j, cov
            cov_frames.append(cov_long)

        # ----------------------------------------------------------------------
        # 5.  concatenate results
        # ----------------------------------------------------------------------
        labels_df = pd.concat(label_frames, ignore_index=True)
        centroids_df = pd.concat(centroid_frames, ignore_index=True)
        cov_df = pd.concat(cov_frames, ignore_index=True)
        scores_df = (
            pd.concat(score_frames, ignore_index=True)
            if score_frames
            else pd.DataFrame()
        )

        return labels_df, centroids_df, cov_df, scores_df

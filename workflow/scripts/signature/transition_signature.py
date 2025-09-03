import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon, cosine
from collections import defaultdict
from itertools import combinations
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

# ------- Distance Functions -------


def dist_func(p, q, method="jsd", eps=1e-12):
    # Convert to numpy arrays
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Apply smoothing and renormalize to make valid probability vectors
    p = (p + eps) / (p.sum() + eps * len(p))
    q = (q + eps) / (q.sum() + eps * len(q))

    if method == "jsd":
        return jensenshannon(p, q)
    elif method == "cosine":
        return cosine(p, q)
    else:
        raise ValueError(f"Unknown distance method: {method}")


def _flat(arr_like):
    # fillna(0) works for DataFrame/Series; np.asarray handles ndarray
    return np.asarray(arr_like.fillna(0).values, dtype=float).ravel()


# ------- Data Filtering and Splitting -------


def filter_by_threshold(df: pd.DataFrame, threshold_days: int) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    user_day_counts = df.groupby("user")["date"].nunique()
    valid_users = user_day_counts[user_day_counts >= threshold_days].index

    return df[df["user"].isin(valid_users)]


def split_chunk(
    df: pd.DataFrame, window: int, id_col: str = "user", split_col=None
) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(by=[id_col, "date"])
    df["day_number"] = df.groupby(id_col).cumcount() + 1

    split_labels = ["split_1", "split_2", "split_3"]
    bins = [0, window, 2 * window, 3 * window, float("inf")]

    if split_col == None:
        # Make 3 splits
        split_labels = ["split_1", "split_2", "split_3"]

        df["split"] = pd.cut(
            df["day_number"],
            bins=bins,
            labels=split_labels + ["rest"],
            right=True,  # include the right edge
            include_lowest=True,  # ensures smallest valid day (1) is included
        )

    else:
        df["split"] = df[split_col]

    df = df[df["split"] != "rest"]
    df = df.drop(columns=["day_number"])

    return df


def transition_matrix(
    df,
    user_col="user",
    date_col="date",
    cluster_col="Cluster",
    states=None,
    normalize=True,
):

    df = df[[user_col, date_col, cluster_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([user_col, date_col])

    df["from_cluster"] = df[cluster_col]
    df["to_cluster"] = df.groupby(user_col, sort=False)[cluster_col].shift(-1)
    pairs = df.dropna(subset=["to_cluster"])

    if states is None:
        states = sorted(pd.unique(df[cluster_col].dropna()))

    mat = (
        pairs.groupby(["from_cluster", "to_cluster"])
        .size()
        .unstack(fill_value=0.0)
        .reindex(index=states, columns=states, fill_value=0.0)
        .astype(float)
    )

    if normalize:
        mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    return mat, pairs


def row_wise_dist(mat_1, mat_2, method="jsd"):
    """
    Compute average row-wise distance between two transition probability matrices.

    Parameters
    ----------
    mat_1, mat_2 : array-like, shape (n, n)
        Row-stochastic transition matrices (rows sum to 1).
    method : str
        Distance metric: "jsd" (default), "cosine", "euclidean", etc.

    Returns
    -------
    float
        Mean distance across rows.
    """
    mat_1 = np.asarray(mat_1, dtype=float)
    mat_2 = np.asarray(mat_2, dtype=float)
    assert mat_1.shape == mat_2.shape, "Matrices must be same shape"

    dists = []
    for i in range(mat_1.shape[0]):
        p = mat_1[i]
        q = mat_2[i]
        d = dist_func(p, q, method)
        dists.append(d)

    return float(np.mean(dists))


def d_self_transition(
    all_mats, splits=["split_1", "split_2", "split_3"], method="jsd"
) -> pd.DataFrame:
    """
    Within-user distances across splits.
    Returns: DataFrame with columns ['user', 'd_self']
    d_self = mean distance between splits (pairwise average).
    """

    users = []
    distances = []

    for user, trans_mat in all_mats.items():
        try:

            mats = [trans_mat[s] for s in splits]
            if len(mats) == 3:
                d12 = row_wise_dist(mats[0], mats[1], method)
                d23 = row_wise_dist(mats[1], mats[2], method)
                # d_self is average of d12 and d23
                d_s = 0.5 * (d12 + d23)
            elif len(mats) == 2:
                d_s = row_wise_dist(mats[0], mats[1], method)
            else:
                raise ValueError(
                    f"d_self_transition only supports 2 or 3 splits, got {len(mats)}"
                )
            users.append(user)
            distances.append(d_s)
        except KeyError:
            continue

    return pd.DataFrame({"user": users, "d_self": distances})


def d_ref_transition(
    all_mats, splits=["split_1", "split_2", "split_3"], method="jsd"
) -> pd.DataFrame:
    """
    Between-user reference distances: match split ik vs split jk.
    Returns: DataFrame with columns ['user_i', 'user_j', 'd_ref']
    d_ref = mean distance across splits (pairwise average).
    """

    users = list(all_mats.keys())
    results = []
    for i, ui in enumerate(users):
        for uj in users[i + 1 :]:
            dists = []
            for s in splits:

                # ensure s exists for both ui and uj (message focuses on uj as requested)
                if s not in all_mats.get(ui, {}):

                    raise KeyError(
                        f"Segment '{s}' not found for ui={ui}. Available: {list(all_mats.get(ui, {}).keys())}"
                    )
                if s not in all_mats.get(uj, {}):

                    print(all_mats[uj])
                    raise KeyError(
                        f"Segment '{s}' not found for uj={uj}. Available: {list(all_mats.get(uj, {}).keys())}"
                    )

                dists.append(row_wise_dist(all_mats[ui][s], all_mats[uj][s], method))

            if len(dists) == len(splits):
                d_ref = sum(dists) / len(dists)
                results.append({"user_i": ui, "user_j": uj, "d_ref": d_ref})

    return pd.DataFrame(results)


# ------- Main Workflow -------


def main(input_fns, output_fns, params):
    logging.info("Loading data...")
    data = pd.read_csv(input_fns[0])

    dist_func = params.dist_method
    study = snakemake.wildcards.study
    threshold_days = int(snakemake.wildcards.window)

    print(f"Distance Method: {dist_func}, study: {study}, threshold: {threshold_days}")

    all_mats = defaultdict(dict)

    # Collect all possible states/clusters to form transition matrix
    states = data.Cluster.unique()

    if study == "globem":
        target_waves = [
            ["INS-W_1", "INS-W_2"],
            ["INS-W_2", "INS-W_3"],
            ["INS-W_3", "INS-W_4"],
        ]

        signatures, dselfs, drefs = {}, {}, {}
        # Create dictionary keys
        keys = ["_".join(t) for t in target_waves]

        for i, target in enumerate(target_waves):

            # users present in all target waves
            users_in_waves = (
                data[data["wave"].isin(target)]
                .drop_duplicates(["user", "wave"])
                .groupby("user")["wave"]
                .nunique()
                .pipe(lambda s: s[s == len(target)].index)
            )

            filtered_clusters_df = data[
                (data.user.isin(users_in_waves)) & (data.wave.isin(target))
            ]
            for (user, wave), g in filtered_clusters_df.groupby(["user", "wave"]):
                mat, _ = transition_matrix(
                    g,
                    user_col="user",
                    date_col="date",
                    cluster_col="Cluster",
                    states=states,
                )

                all_mats[user][wave] = mat

            # Get transition mat for current users
            current_mat = {k: all_mats[k] for k in users_in_waves}

            logging.info(f"Processing waves: {target} with {len(users_in_waves)} users")

            logging.info("Calculating self-distances...")
            d_self_df = d_self_transition(current_mat, splits=target, method=dist_func)

            logging.info("Calculating reference distances...")
            d_ref_df = d_ref_transition(current_mat, splits=target, method=dist_func)

            # Create key
            dselfs[keys[i]] = d_self_df
            drefs[keys[i]] = d_ref_df

        # Combine all dselfs and drefs into DataFrames

        all_mats_df = (
            pd.concat(
                {
                    pid: pd.concat(splits, names=["wave"])
                    for pid, splits in all_mats.items()
                },
                names=["id"],
            )
            .stack()
            .reset_index(name="value")
        )
        combined_dself = (
            pd.concat(dselfs, names=["wave"], keys=dselfs.keys())
            .reset_index(level=0)
            .rename(columns={"level_0": "wave"})
        )
        combined_dref = (
            pd.concat(drefs, names=["wave"], keys=drefs.keys())
            .reset_index(level=0)
            .rename(columns={"level_0": "wave"})
        )

        # Save to output files
        all_mats_df.to_csv(output_fns[0], index=False)
        combined_dself.to_csv(output_fns[1], index=False)
        combined_dref.to_csv(output_fns[2])
    else:

        logging.info("Filtering data by threshold...")
        data = filter_by_threshold(data, threshold_days)

        logging.info("Splitting data into chunks...")
        window = int(threshold_days / 3)
        data = split_chunk(data, window=window, id_col="user")

        logging.info("Calculating transition matrix...")
        for (user, split), g in data.groupby(["user", "split"], sort=False):
            mat, _ = transition_matrix(
                g,
                user_col="user",
                date_col="date",
                cluster_col="Cluster",
                states=states,
            )
            all_mats[user][split] = mat

        logging.info("Calculating self-distances...")
        d_self_df = d_self_transition(all_mats, splits=params.splits, method=dist_func)

        logging.info("Calculating reference distances...")
        d_ref_df = d_ref_transition(all_mats, splits=params.splits, method=dist_func)

        logging.info("Saving results...")

        all_mats_df = (
            pd.concat(
                {
                    pid: pd.concat(splits, names=["split"])
                    for pid, splits in all_mats.items()
                },
                names=["id"],
            )
            .stack()
            .reset_index(name="value")
        )
        all_mats_df.to_csv(output_fns[0], index=False)
        d_self_df.to_csv(output_fns[1], index=False)
        d_ref_df.to_csv(output_fns[2])

        logging.info("Processing complete.")


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

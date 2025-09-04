import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon, cosine
from collections import defaultdict
from itertools import combinations
import os
import logging
from typing import Callable, List, Optional, Tuple

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


# ------- Data Filtering and Splitting -------


def filter_by_threshold(df: pd.DataFrame, threshold_days: int) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    user_day_counts = df.groupby("user", observed=True)["date"].nunique()
    valid_users = user_day_counts[user_day_counts >= threshold_days].index

    return df[df["user"].isin(valid_users)]


def split_chunk(
    df: pd.DataFrame,
    window: int,
    splits: Optional[List[str]] = None,
    id_col: str = "user",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Split each individual's timeline into fixed-size windows and label them.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns [id_col, date_col].
    window : int
        Number of days per split window.
    splits : list[str]
        Labels for the windows (e.g., ["split_1", "split_2", "split_3"]).
        Rows beyond the last full window are labeled 'rest' and removed.
    id_col : str
        ID column name.
    date_col : str
        Date/timestamp column name.

    Returns
    -------
    pd.DataFrame
        Input rows with an added 'split' column, excluding 'rest'.
    """
    if splits is None:
        raise ValueError(
            "Provide explicit split labels, e.g., ['split_1','split_2','split_3']."
        )
    if window <= 0:
        raise ValueError("`window` must be a positive integer.")
    if len(splits) < 1:
        raise ValueError("`splits` must contain at least one label.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(by=[id_col, date_col])
    df["day_number"] = df.groupby(id_col).cumcount()  # 0-based within each id

    k = len(splits)
    # Build bin edges: [-1, w-1, 2w-1, ..., kw-1, inf]
    bin_edges = [-1] + [i * window - 1 for i in range(1, k + 1)] + [float("inf")]
    labels = splits + ["rest"]

    # Assign split labels per id (vectorized; day_number already per-id)
    df["split"] = pd.cut(df["day_number"], bins=bin_edges, labels=labels, right=True)

    # Keep only the requested k splits; drop the trailing remainder
    out = (
        df[df["split"].isin(splits)].drop(columns=["day_number"]).reset_index(drop=True)
    )
    return out


# ------- Compute transition matrix -------


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
    df["to_cluster"] = df.groupby(user_col, sort=False, observed=True)[
        cluster_col
    ].shift(-1)
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


# ------- Transition signature -------
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


####### MAIN #######
def process_signature(df, threshold_days, splits, dist_func):

    all_mats = defaultdict(dict)
    df = filter_by_threshold(df, threshold_days)

    # Collect all possible states/clusters to form transition matrix
    states = df.Cluster.unique()

    counts = df.groupby("user", observed=True).size()
    print(
        f"min obs/user = {counts.min()}, max obs/user = {counts.max()} (n_users={counts.size})"
    )
    window = threshold_days // len(splits)
    df = split_chunk(df, window=window, splits=splits, id_col="user")

    logging.info("Calculating transition matrix...")
    for (user, split), g in df.groupby(["user", "split"], sort=False):
        mat, _ = transition_matrix(
            g,
            user_col="user",
            date_col="date",
            cluster_col="Cluster",
            states=states,
        )
        all_mats[user][split] = mat

    ds = d_self_transition(all_mats, splits=splits, method=dist_func)
    dr = d_ref_transition(all_mats, splits=splits, method=dist_func)

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

    return all_mats_df, ds, dr


def run_pipeline(data, study, threshold_days, splits, dist_func, output_fns):
    one = lambda df: process_signature(df, threshold_days, splits, dist_func)

    if study == "globem" and "wave" in data.columns:
        sig_parts, dself_parts, dref_parts = [], [], []
        for wave, sample in data.groupby("wave", sort=True, observed=True):

            us, ds, dr = one(sample)
            us["wave"] = wave
            ds["wave"] = wave
            dr["wave"] = wave
            sig_parts.append(us)
            dself_parts.append(ds)
            dref_parts.append(dr)

        user_signature = pd.concat(sig_parts)
        d_self_df = pd.concat(dself_parts)
        d_ref_df = pd.concat(dref_parts)
    else:
        user_signature, d_self_df, d_ref_df = one(data)

    user_signature.to_csv(output_fns[0], index=False)
    d_self_df.to_csv(output_fns[1], index=False)
    d_ref_df.to_csv(output_fns[2])


def main(input_fns, output_fns, params):
    logging.info("Loading data...")
    data = pd.read_csv(input_fns[0])

    dist_func = params.dist_method
    study = snakemake.wildcards.study
    threshold_days = int(snakemake.wildcards.window)
    splits = (
        params.splits
    )  # e.g., ["split_1","split_2"] or ["split_1","split_2","split_3"]

    print(f"Distance Method: {dist_func}, study: {study}")
    run_pipeline(data, study, threshold_days, splits, dist_func, output_fns)


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

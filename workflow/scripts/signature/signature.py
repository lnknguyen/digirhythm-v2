import numpy as np
import pandas as pd
import logging
from scipy.spatial.distance import jensenshannon, cosine
from itertools import combinations
from typing import Callable, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def dist_func(p, q, method="jsd"):
    """
    Generalized distance function.
    Supports 'jsd' (Jensen-Shannon) and 'cosine'
    """
    p = np.asarray(p)
    q = np.asarray(q)

    if method == "jsd":
        return jensenshannon(p, q, base=2)
    elif method == "cosine":
        return cosine(p, q)
    else:
        raise ValueError(f"Unknown distance method: {method}")


def filter_by_threshold(df: pd.DataFrame, threshold_days: int) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    user_day_counts = df.groupby("user")["date"].nunique()
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
    df["day_number"] = df.groupby(id_col, observed=True).cumcount()  # 0-based within each id

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


def signature(df: pd.DataFrame, ranked: bool) -> pd.DataFrame:

    # Assert: dataset must contain a 'split' column
    assert "split" in df.columns, "Expected column 'split' in `df`."

    user_signature = (
        df.groupby(["user", "split", "Cluster"], observed=True).size().reset_index(name="count")
    )

    user_signature["percentage"] = user_signature.groupby(["user", "split"], observed=True)[
        "count"
    ].transform(lambda x: 100 * x / x.sum())

    user_signature = user_signature.sort_values(
        ["user", "split", "Cluster"]
    ).reset_index(drop=True)

    if ranked:
        user_signature = (
            user_signature.sort_values(
                ["user", "split", "percentage"], ascending=[True, True, False]
            )
            # .groupby(["user", "split"])
            # .head(5)
            # .reset_index(drop=True)
        )

        user_signature["rank"] = (
            user_signature.groupby(["user", "split"], observed=True).cumcount() + 1
        )

        user_signature = user_signature.pivot_table(
            index=["user", "split"], columns="rank", values="percentage", fill_value=0,observed=True
        )
    else:
        user_signature = user_signature.pivot_table(
            index=["user", "split"],
            columns="Cluster",
            values="percentage",
            fill_value=0,
            observed=True
        )

    return user_signature


def d_self(
    signature_df: pd.DataFrame,
    splits=["split_1", "split_2", "split_3"],
    method: str = "jsd",
) -> pd.DataFrame:
    users = []
    distances = []

    for user, group in signature_df.groupby(level="user"):
        try:

            if len(splits) == 3:
                dist1 = group.loc[(user, splits[0])].values
                dist2 = group.loc[(user, splits[1])].values
                dist3 = group.loc[(user, splits[2])].values

                d_s = 0.5 * (
                    dist_func(dist1, dist2, method) + dist_func(dist2, dist3, method)
                )
            elif len(splits) == 2:
                dist1 = group.loc[(user, splits[0])].values
                dist2 = group.loc[(user, splits[1])].values

                d_s = dist_func(dist1, dist2, method)
            else:
                raise ValueError(
                    f"d_self only supports exactly 2 or 3 splits, got {len(splits)}"
                )

            users.append(user)
            distances.append(d_s)
        except KeyError:
            continue

    return pd.DataFrame({"user": users, "d_self": distances})


def d_ref(
    signature_df: pd.DataFrame,
    splits=["split_1", "split_2", "split_3"],
    method: str = "jsd",
    return_: str = "full",
) -> pd.DataFrame:
    """
    Build a user×user distance matrix by averaging distances across the given splits.
    Returns the upper triangle by default (NaNs below the diagonal).

    Parameters
    ----------
    df : pd.DataFrame
        Indexed by a MultiIndex with levels ('user', 'split').
    splits : sequence of str
        Split labels to include; all must exist for each user pair.
    method : str
        Distance method passed to dist_func.
    return_ : {"triu_df", "condensed", "full"}
        - "triu_df": DataFrame with only upper triangle (k=1) retained.
        - "condensed": Series of upper-triangle values with (user_i, user_j) index.
        - "full": full symmetric DataFrame.

    """
    users = signature_df.index.get_level_values("user").unique()
    d_ref_df = pd.DataFrame(index=users, columns=users, dtype=float)

    for i, j in combinations(users, 2):
        dists = []
        for s in splits:
            try:
                di = signature_df.loc[(i, s)].values
                dj = signature_df.loc[(j, s)].values
            except KeyError:
                dists = []  # missing split for this pair -> skip
                break
            dists.append(dist_func(di, dj, method))

        if len(dists) == len(splits):
            d_r = sum(dists) / len(splits)
            d_ref_df.loc[i, j] = d_r
            d_ref_df.loc[j, i] = d_r

    if return_ == "full":
        return d_ref_df

    if return_ == "condensed":
        A = d_ref_df.to_numpy()
        iu, ju = np.triu_indices_from(A, k=1)
        idx = pd.MultiIndex.from_arrays(
            [users.take(iu), users.take(ju)], names=["user_i", "user_j"]
        )
        return pd.Series(A[iu, ju], index=idx, name="d_ref").dropna()

    # default: upper triangle DataFrame (NaN below diagonal, no diagonal)
    mask = np.triu(np.ones(d_ref_df.shape, dtype=bool), k=1)
    return d_ref_df.where(mask)


####### MAIN #######
def process_signature(df, threshold_days, splits, ranked, dist_func):
    df = filter_by_threshold(df, threshold_days)

    counts = df.groupby("user").size()
    print(
        f"min obs/user = {counts.min()}, max obs/user = {counts.max()} (n_users={counts.size})"
    )
    window = threshold_days // len(splits)
    df = split_chunk(df, window=window, splits=splits, id_col="user")
    us = signature(df, ranked)
    ds = d_self(us, splits=splits, method=dist_func)
    dr = d_ref(us, splits=splits, method=dist_func)
    return us, ds, dr


def run_pipeline(data, study, threshold_days, splits, ranked, dist_func, output_fns):
    one = lambda df: process_signature(df, threshold_days, splits, ranked, dist_func)

    if study == "globem" and "wave" in data.columns:
        sig_parts, dself_parts, dref_parts = [], [], []
        for wave, sample in data.groupby("wave", sort=True):

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

    ranked = params.ranked == "ranked"
    dist_func = params.dist_method
    study = snakemake.wildcards.study
    threshold_days = int(snakemake.wildcards.window)
    splits = (
        params.splits
    )  # e.g., ["split_1","split_2"] or ["split_1","split_2","split_3"]

    print(f"Ranked: {ranked}, Distance Method: {dist_func}, study: {study}")
    run_pipeline(data, study, threshold_days, splits, ranked, dist_func, output_fns)


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

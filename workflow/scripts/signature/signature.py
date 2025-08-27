import numpy as np
import pandas as pd
import logging
from scipy.spatial.distance import jensenshannon, cosine
from itertools import combinations

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


def split_chunk(df: pd.DataFrame, window: int, id_col: str = "user") -> pd.DataFrame:

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(by=[id_col, "date"])
    df["day_number"] = df.groupby(id_col).cumcount()

    # Make 3 splits. 90 days each
    split_labels = ["split_1", "split_2", "split_3"]

    # Create a split column to annotate split
    df["split"] = pd.cut(
        df["day_number"],
        bins=[-1, window - 1, 2 * window - 1, 3 * window - 1, float("inf")],
        labels=split_labels + ["rest"],
    )

    df = df[df["split"] != "rest"]
    df = df.drop(columns=["day_number"])

    return df


def signature(df: pd.DataFrame, ranked: bool) -> pd.DataFrame:

    # Assert: dataset must contain a 'split' column
    assert "split" in df.columns, "Expected column 'split' in `df`."

    user_signature = (
        df.groupby(["user", "split", "Cluster"]).size().reset_index(name="count")
    )

    user_signature["percentage"] = user_signature.groupby(["user", "split"])[
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
            user_signature.groupby(["user", "split"]).cumcount() + 1
        )

        user_signature = user_signature.pivot_table(
            index=["user", "split"], columns="rank", values="percentage", fill_value=0
        )
    else:
        user_signature = user_signature.pivot_table(
            index=["user", "split"],
            columns="Cluster",
            values="percentage",
            fill_value=0,
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


def main(input_fns, output_fns, params):
    logging.info("Loading data...")
    data = pd.read_csv(input_fns[0])

    ranked = True if params.ranked == "ranked" else False
    dist_func = params.dist_method
    study = snakemake.wildcards.study

    print(
        f"Ranked: {ranked}, Distance Method: {dist_func}, study: {snakemake.wildcards.study}"
    )

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

            logging.info(f"Processing waves: {target} with {len(users_in_waves)} users")

            # Assign wave to 'split', as split
            # column will be used to compute signature
            filtered_clusters_df["split"] = filtered_clusters_df["wave"]

            logging.info("Calculating user signatures...")
            signature_df = signature(filtered_clusters_df, ranked)

            logging.info("Calculating self-distances...")
            d_self_df = d_self(signature_df, splits=target, method=dist_func)

            logging.info("Calculating reference distances...")
            d_ref_df = d_ref(signature_df, splits=target, method=dist_func)

            # Create key
            dselfs[keys[i]] = d_self_df
            drefs[keys[i]] = d_ref_df
            signatures[keys[i]] = signature_df

        # Combine all dselfs and drefs into DataFrames
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
        combined_signatures = (
            pd.concat(signatures, names=["wave"], keys=signatures.keys())
            .reset_index(level=0)
            .rename(columns={"level_0": "wave"})
        )

        # Save to output files
        combined_signatures.to_csv(output_fns[0])
        combined_dself.to_csv(output_fns[1], index=False)
        combined_dref.to_csv(output_fns[2])
    else:
        logging.info("Filtering data by threshold...")
        data = filter_by_threshold(data, params.threshold_days)

        logging.info("Splitting data into chunks...")
        window = int(params.threshold_days / 3)
        data = split_chunk(data, window=window, id_col="user")

        logging.info("Calculating user signatures...")
        user_signature = signature(data, ranked)

        logging.info("Calculating self-distances...")
        d_self_df = d_self(user_signature, splits=params.splits, method=dist_func)

        logging.info("Calculating reference distances...")
        d_ref_df = d_ref(user_signature, splits=params.splits, method=dist_func)

        logging.info("Saving results...")
        user_signature.to_csv(output_fns[0])
        d_self_df.to_csv(output_fns[1], index=False)
        d_ref_df.to_csv(output_fns[2])

        logging.info("Processing complete.")


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

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
    Supports 'jsd' (Jensen-Shannon) and 'l2' (Euclidean/ℓ2 norm).
    """
    p = np.asarray(p)
    q = np.asarray(q)

    if method == "jsd":
        return jensenshannon(p, q, base=2)
    elif method == "cosine":
        return cosine(p, q)
    else:
        raise ValueError(f"Unknown distance method: {method}")


def filter_by_threshold(df: pd.DataFrame, threshold_days: int = 30 * 9) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    user_day_counts = df.groupby("user")["date"].nunique()
    valid_users = user_day_counts[user_day_counts >= threshold_days].index

    return df[df["user"].isin(valid_users)]


def split_chunk(
    df: pd.DataFrame, window: int = 90, id_col: str = "user"
) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(by=[id_col, "date"])
    df["day_number"] = df.groupby(id_col).cumcount()

    split_labels = ["split_1", "split_2", "split_3"]
    df["split"] = pd.cut(
        df["day_number"],
        bins=[-1, window - 1, 2 * window - 1, 3 * window - 1, float("inf")],
        labels=split_labels + ["rest"],
    )

    df = df[df["split"] != "rest"]
    df = df.drop(columns=["day_number"])

    return df


def signature(df: pd.DataFrame, ranked: bool) -> pd.DataFrame:
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
            .groupby(["user", "split"])
            .head(5)
            .reset_index(drop=True)
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


def d_self(df: pd.DataFrame, method: str = "jsd") -> pd.DataFrame:
    users = []
    distances = []

    for user, group in df.groupby(level="user"):
        try:
            dist1 = group.loc[(user, "split_1")].values
            dist2 = group.loc[(user, "split_2")].values
            dist3 = group.loc[(user, "split_3")].values

            d_s = 0.5 * (
                dist_func(dist1, dist2, method) + dist_func(dist2, dist3, method)
            )
            users.append(user)
            distances.append(d_s)
        except KeyError:
            continue

    return pd.DataFrame({"user": users, "d_self": distances})


def d_ref(df: pd.DataFrame, method: str = "jsd") -> pd.DataFrame:
    users = df.index.get_level_values("user").unique()
    d_ref_df = pd.DataFrame(index=users, columns=users, dtype=float)

    for i, j in combinations(users, 2):
        try:
            d_i1 = df.loc[(i, "split_1")].values
            d_j1 = df.loc[(j, "split_1")].values

            d_i2 = df.loc[(i, "split_2")].values
            d_j2 = df.loc[(j, "split_2")].values

            d_i3 = df.loc[(i, "split_3")].values
            d_j3 = df.loc[(j, "split_3")].values

            d_r = (
                dist_func(d_i1, d_j1, method)
                + dist_func(d_i2, d_j2, method)
                + dist_func(d_i3, d_j3, method)
            ) / 3

            d_ref_df.loc[i, j] = d_r
            d_ref_df.loc[j, i] = d_r
        except KeyError:
            continue

    return d_ref_df


def main(input_fns, output_fns, params):
    logging.info("Loading data...")
    data = pd.read_csv(input_fns[0])

    ranked = True if params.ranked == "ranked" else False
    if ranked == True:

        dist_method = "jsd"
    else:

        dist_method = "cosine"

    logging.info(f"Processing with ranked={ranked}, method={dist_method}")

    filtered_data = filter_by_threshold(data, threshold_days=270)
    filtered_data = split_chunk(filtered_data, window=90)

    signature_df = signature(filtered_data, ranked=ranked)

    logging.info("Computing d_self...")
    dself = d_self(signature_df, method=dist_method)

    logging.info("Computing d_ref...")
    dref = d_ref(signature_df, method=dist_method)

    logging.info("Saving outputs...")
    signature_df.to_csv(output_fns.signature)
    dself.to_csv(output_fns.d_self, index=False)
    dref.to_csv(output_fns.d_ref)

    logging.info("Done.")


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

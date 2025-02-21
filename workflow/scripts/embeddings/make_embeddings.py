import pandas as pd
import umap

SEED = 2508


def main(input_fns, output_fns, params, wildcards):
    # Load data
    data = pd.read_csv(input_fns[0])

    # Retrieve feature list and grouping
    groupby = params["groupby"][wildcards.study]
    features = params["features"]
    cols = groupby + features

    # Keep only groupby and features columns
    df = data[cols]

    # Normalize features per user
    # Using z-score normalization. If the std is zero, use 1 to avoid division by zero.
    def safe_normalize(x):
        std = x.std()
        return (x - x.mean()) / (std if std != 0 else 1)

    df_norm = df.copy()

    df_norm[features] = df_norm.groupby(["user"])[features].transform(safe_normalize)
    df_norm.dropna(subset=features, inplace=True)

    # Initialize UMAP with 2 components
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=SEED)
    components = reducer.fit_transform(df_norm[features])

    pd.DataFrame(components, columns=["UMAP_1", "UMAP_2"]).to_csv(output_fns[0])


if __name__ == "__main__":

    main(snakemake.input, snakemake.output, snakemake.params, snakemake.wildcards)

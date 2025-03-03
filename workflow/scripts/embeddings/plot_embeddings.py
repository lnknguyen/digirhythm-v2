import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SEED = 2508


def main(input_fns, output_fns, wildcards):

    embeddings = pd.read_csv(input_fns[0], index_col=0)
    labels = pd.read_csv(input_fns[1], index_col=0)

    # Plot embeddings with labels
    combined_df = pd.concat([embeddings, labels], axis=1)

    print(combined_df.head())

    # Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=combined_df.iloc[:, 0],
        y=combined_df.iloc[:, 1],
        hue=combined_df["Cluster"],
        palette="Set2",
        s=70,
        edgecolor="k",
    )
    plt.title(f"UMAP Embeddings with {wildcards.algo} cluster")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Cluster")
    plt.tight_layout()

    # Save plot
    plt.savefig(output_fns[0])
    plt.close()


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.wildcards)

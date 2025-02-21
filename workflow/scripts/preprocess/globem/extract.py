import pandas as pd


def main(input_fns, output_fns, params):

    res = []

    for fn in input_fns:
        df = pd.read_csv(fn, index_col=0)

        # Extract wave name from the file path if needed
        wave = fn.split("/")[-3]  # Assuming wave is in the second last directory
        print(f"Processing Wave: {wave} for Sensor: {snakemake.wildcards.sensor}")

        # Rename
        df.rename(columns={"pid": "user"}, inplace=True)
        df["wave"] = wave
        res.append(df)

    # Output
    res = pd.concat(res)
    res.to_csv(output_fns[0])


if __name__ == "__main__":

    input_fns = snakemake.input
    output_fns = snakemake.output
    params = snakemake.params

    main(input_fns, output_fns, params)

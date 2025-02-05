import argparse
import pandas as pd
import sys

import logging


def concatenate_features(input_paths):
    """
    Concatenate multiple feature files into a single DataFrame and save it.

    Args:
        input_paths (list): List of input file paths.
        output_path (str): Path to save the concatenated DataFrame.
    """
    out = []

    for path in input_paths:

        df = pd.read_csv(path)
        out.append(df)

    concatenated_df = pd.concat(out)

    return concatenated_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fns",
        type=str,
        nargs="+",
        help="Input files (full path). Can use *. If more than one, loop over them",
        required=True,
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        nargs="+",
        help="Output files (full path)",
        required=True,
    )

    args = parser.parse_args()

    # Call the concatenate_features function
    df = concatenate_features(args.input_fns)

    df.to_csv(args.output_fn[0])

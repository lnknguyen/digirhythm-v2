import numpy as np
import pandas as pd
import os
import logging
from functools import reduce

RENAME_COLS = {
    # Activity
    ("step:steps:night", "magnitude_std:night"): "activity_night",
    ("step:steps:morning", "magnitude_std:morning"): "activity_morning",
    ("step:steps:afternoon", "magnitude_std:afternoon"): "activity_afternoon",
    ("step:steps:evening", "magnitude_std:evening"): "activity_evening",
    ("step:allday", "magnitude_std:allday"): "activity_allday",
    # Call
    "call:total_duration:night": "call_total_duration_night",
    "call:total_duration:morning": "call_total_duration_morning",
    "call:total_duration:afternoon": "call_total_duration_afternoon",
    "call:total_duration:evening": "call_total_duration_evening",
    "call:total_duration:allday": "call_total_duration_allday",
    # Screen
    (
        "screen:screen_use_durationtotal:night",
        "f_screen:phone_screen_rapids_sumdurationunlock:night",
    ): "screen_night",
    (
        "screen:screen_use_durationtotal:morning",
        "f_screen:phone_screen_rapids_sumdurationunlock:morning",
    ): "screen_morning",
    (
        "screen:screen_use_durationtotal:afternoon",
        "f_screen:phone_screen_rapids_sumdurationunlock:afternoon",
    ): "screen_afternoon",
    (
        "screen:screen_use_durationtotal:evening",
        "f_screen:phone_screen_rapids_sumdurationunlock:evening",
    ): "screen_evening",
    (
        "screen:screen_use_durationtotal:allday",
        "f_screen:phone_screen_rapids_sumdurationunlock:allday",
    ): "screen_allday",
    # Sleep
    ("sleep_start_hhmm"): "sleep_onset",
    ("sleep_end_hhmm"): "sleep_offset",
    # ("midsleep_hhmm"): "midsleep",
}


def rename_columns(df):
    """
    Rename DataFrame columns using a many-to-one mapping.

    Args:
        df (pd.DataFrame): Input DataFrame.
        mapping (dict): Dictionary where keys are tuples of column names to replace,
                        and values are the new column names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """

    # Create a new dictionary with expanded keys
    expanded_RENAME_COLS = {}

    for key, value in RENAME_COLS.items():
        if isinstance(key, tuple):  # Check if the key is a tuple
            for subkey in key:
                expanded_RENAME_COLS[subkey] = value
        else:
            expanded_RENAME_COLS[key] = value

    return df.rename(columns=expanded_RENAME_COLS, inplace=False)


def adapter(input_fns):
    """
    Read in the files and convert columns name
    """

    dfs = []
    for fn in input_fns:

        df = pd.read_csv(fn, index_col=0)

        # Get rid of index column, if exists
        if "index" in df.columns:
            df = df.drop(columns=["index"])

        dfs.append(df)

    return dfs


def concatenate_features(dfs, groupby):
    """
    Concatenate multiple feature files into a single DataFrame and save it.

    Args:
        dfs (list): List of DataFrames.
    """

    merged_df = dfs[0]

    for df in dfs[1:]:

        merged_df = pd.merge(merged_df, df, on=groupby, how="outer")

    return merged_df


def _log_transform(df, features, suffix="_log1p", inplace=False):

    transformed = {}
    for col in features:
        if (df[col] < 0).any():
            raise ValueError(
                f"Feature '{col}' contains negative values; "
                "log1p is intended for non-negative data."
            )
        transformed[col] = np.log1p(df[col])

    if inplace:
        for col in features:
            df[col] = transformed[col]
    else:
        for col in features:
            df[f"{col}{suffix}"] = transformed[col]

    return df


def main(input_fns, output_fn, params, wildcards):

    groupby = params["groupby"][wildcards.study]

    dfs = adapter(input_fns)
    df = concatenate_features(dfs, groupby)

    # Convert columns name
    df = rename_columns(df)

    # Convert call features to log
    call_features = [
        "call_total_duration_night",
        "call_total_duration_morning",
        "call_total_duration_afternoon",
        "call_total_duration_evening",
        "call_total_duration_allday",
    ]

    # Safety check
    existing_call_features = [col for col in call_features if col in df.columns]

    if existing_call_features:
        df = _log_transform(df, existing_call_features)
    

    # Save
    df.to_csv(output_fn)


if __name__ == "__main__":

    main(snakemake.input, snakemake.output[0], snakemake.params, snakemake.wildcards)

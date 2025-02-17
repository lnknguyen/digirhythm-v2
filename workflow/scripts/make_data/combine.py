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
    "screen:screen_use_durationtotal:night": "screen_night",
    "screen:screen_use_durationtotal:morning": "screen_morning",
    "screen:screen_use_durationtotal:afternoon": "screen_afternoon",
    "screen:screen_use_durationtotal:evening": "screen_evening",
    "screen:screen_use_durationtotal:allday": "screen_allday",
    # Sleep
    ("sleep_start_hhmm"): "sleep_onset",
    ("sleep_end_hhmm"): "sleep_offset",
    #("midsleep_hhmm"): "midsleep",
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

        df = pd.read_csv(fn)
        # Add group column if not exist
        if "group" not in df.columns:
            df["group"] = "None"

        dfs.append(df)

    return dfs


def concatenate_features(dfs):
    """
    Concatenate multiple feature files into a single DataFrame and save it.

    Args:
        dfs (list): List of DataFrames.
    """

    merged_df = dfs[0]
    
    for df in dfs[1:]:
        
        merged_df = pd.merge(merged_df, df, on=["user", "date", "group"], how="outer")


    return merged_df

def main(input_fns, output_fn):

    
    dfs = adapter(input_fns)
    df = concatenate_features(dfs)

    # Convert columns name
    df = rename_columns(df)

    #Test
    cols_to_keep = RENAME_COLS.values()
    df.to_csv(output_fn)


if __name__ == "__main__":

    main(snakemake.input, snakemake.output[0])

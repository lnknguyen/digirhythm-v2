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
    "screen:screen_on_durationtotal:night": "screen_night",
    "screen:screen_on_durationtotal:morning": "screen_morning",
    "screen:screen_on_durationtotal:afternoon": "screen_afternoon",
    "screen:screen_on_durationtotal:evening": "screen_evening",
    "screen:screen_on_durationtotal:allday": "screen_allday",
    # Sleep
    ("sleep_start_hhmm"): "sleep_onset",
    ("sleep_end_hhmm"): "sleep_offset",
    ("midsleep_hhmm"): "midsleep",
}


def _rename_columns(df, mapping):
    """
    Rename DataFrame columns using a many-to-one mapping.

    Args:
        df (pd.DataFrame): Input DataFrame.
        mapping (dict): Dictionary where keys are tuples of column names to replace,
                        and values are the new column names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    new_columns = []
    for col in df.columns:
        # Check if the column matches any key in the mapping
        replaced = False
        for key_tuple, new_name in mapping.items():
            if col in key_tuple:
                new_columns.append(new_name)
                replaced = True
                break
        if not replaced:
            new_columns.append(col)  # Keep original name if no match
    df.columns = new_columns
    return df


def adapter(input_fns):
    """
    Read in the files and convert columns name
    """

    dfs = []
    for fn in input_fns:

        df = pd.read_csv(fn)
        print(df.columns)
        # Get the first part of the file name (sensor)
        sensor = os.path.basename(fn).split("_")[0]

        # Convert columns name
        df = _rename_columns(df, mapping=RENAME_COLS)

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

    # Merge by 'user', 'date'
    merged_df = dfs[0]

    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=["user", "date", "group"], how="outer")

    return merged_df


def main(input_fns, output_fn):

    dfs = adapter(input_fns)
    df = concatenate_features(dfs)

    df.to_csv(output_fn)


if __name__ == "__main__":

    main(snakemake.input, snakemake.output[0])

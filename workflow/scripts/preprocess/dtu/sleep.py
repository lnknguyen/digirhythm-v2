from .base import BaseProcessor
from dataclasses import dataclass, field
import niimpy
import pandas as pd
import niimpy.preprocessing.screen as screen
import numpy as np
import polars as pl

import os

path = os.path.abspath(niimpy.__file__)
print(path)


@dataclass
class SleepProcessor(BaseProcessor):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

    def longest_inactive_sequence(self, series):
        """
        Identify the longest sequence of inactive periods (0s) in a series,
        combining sequences separated by a single active period (1).

        Parameters:
        series (list or np.array): A series of binary values representing activity
                                   status, where 0 indicates inactivity and 1 indicates activity.

        Returns:
        tuple: A tuple containing:
               - max_len (int): The length of the longest sequence of 0s.
               - max_start_idx (int): The starting index of the longest sequence of 0s.
               - max_end_idx (int): The ending index of the longest sequence of 0s.
        """
        # Initialize variables to track the longest sequence
        max_len = 0
        seq_len = []
        current_len = 0
        start_idx = []
        end_idx = []
        in_seq = True

        for i, value in enumerate(series):
            if value == 0:
                if current_len == 0:
                    start_idx.append(i)
                current_len += 1
            else:
                if current_len > 0:
                    end_idx.append(i - 1)
                    seq_len.append(current_len)
                current_len = 0

        # Check if inactivity sequence is separated by one activity
        arr_len = len(end_idx) - 1
        for i in range(arr_len):
            gap = start_idx[i + 1] - end_idx[i]
            if gap == 1:

                # Combine sequence
                seq_len[i : i + 1] = seq_len[i] + seq_len[i + 1]

                # Pop start and end
                start_idx.pop(i + 1)
                end_idx.pop(i)

        if len(seq_len) > 0:
            max_idx = np.argmax(seq_len)
            return seq_len[max_idx], start_idx[max_idx], end_idx[max_idx]
        else:
            return (None, None, None)

    def compute_sleep_properties(self, df):

        # Compute midsleep
        df["midsleep"] = df["sleep_start"] + (df["sleep_end"] - df["sleep_start"]) / 2

        # Function to convert time to hh.mm format with adjustment for times past midnight
        def convert_to_hhmm(t):
            hour = t.hour
            if hour < 12:
                hour += 24
            return hour + t.minute / 60

        # Compute sleep_start_hhmm, sleep_end_hhmm, and midsleep_hhmm
        df["sleep_start_hhmm"] = df["sleep_start"].apply(convert_to_hhmm)
        df["sleep_end_hhmm"] = df["sleep_end"].apply(convert_to_hhmm)
        df["midsleep_hhmm"] = df["midsleep"].apply(convert_to_hhmm)
        df["sleep_duration"] = df["longest_inactive_seq"] * 5 / 60

        return df

    def filter_artifacts(self, df):
        df = df[(df.sleep_duration > 3) & (df.sleep_duration < 13)]
        return df

    def extract_features(self) -> pd.DataFrame:

        empty_batt_data = pd.DataFrame()

        rule = "5min"
        wrapper_features = {
            screen.screen_count: {
                "screen_column_name": "screen_status",
                "resample_args": {"rule": rule},
            }
        }

        def resample_data(df, rule="5min") -> pd.DataFrame:

            df_resampled = df.resample(rule)["screen_status"].sum().ffill()
            return df_resampled

        df = (
            self.data.pipe(self.convert_copenhagen_time)
            .pipe(self.set_datetime_index)
            .pipe(self.remove_first_last_day)
            .pipe(self.drop_duplicates_and_sort)
            .groupby(["user", "device"])
            .pipe(resample_data)
            .reset_index()
        )

        # Reset datetime index
        df.set_index("level_2", inplace=True)
        print(df.head())

        # Adjust datetime for custom day starting at 3 PM
        df["date"] = (df.index - pd.Timedelta(hours=15)).date

        # Group by user and custom day
        grouped = df.groupby(["user", "date"])

        # Store results
        results = []

        for (user, day), group in grouped:

            # Calculate longest sequence of 0s
            longest_inactive_sequence, start_idx, end_idx = (
                self.longest_inactive_sequence(group["screen_status"].values)
            )

            # Get the start and end datetime values
            sleep_start = group.index[start_idx] if start_idx is not None else None
            sleep_end = group.index[end_idx] if end_idx is not None else None

            # Append result
            results.append(
                {
                    "user": user,
                    "date": day,
                    "longest_inactive_seq": longest_inactive_sequence,
                    "sleep_start": sleep_start,
                    "sleep_end": sleep_end,
                }
            )

        prefixes = [
            "sleep_duration",
            "sleep_start_hhmm",
            "sleep_end_hhmm",
            "midsleep_hhmm",
        ]

        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = (
            results_df.pipe(self.compute_sleep_properties)
            .pipe(self.filter_artifacts)
            .pipe(
                self.normalize_within_user, prefixes=prefixes
            )  # normalize within-user features
            .pipe(
                self.normalize_between_user, prefixes=prefixes
            )  # normalize within-user features
        )

        return results_df

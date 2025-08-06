import sys
import os

sys.path.append("../../")

import niimpy
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from util import progress_decorator


@dataclass
class BaseProcessor:
    """
    BaseProcessor is the base class for all sensor processing classes. It provides a structured
    way to handle sensor data processing with support for different frequencies of data
    aggregation and summarization.

    Attributes:elif
        path (str): The file system path where sensor data files are located.
        group (str): The name of the data grouping to apply, typically based on sensor ID or location.
        data (pd.DataFrame): The data frame containing the sensor data. Defaults to an empty DataFrame.
        frequency (str): Defines the frequency for data aggregation and summarization. It determines
                         how the sensor data is processed and transformed. The possible values are:
            - '4epochs': Divides each day into four time periods (epochs). Each epoch represents
                         a specific part of the day:
                * Night: 00:00 - 05:59
                * Morning: 06:00 - 11:59
                * Afternoon: 12:00 - 17:59
                * Evening: 18:00 - 23:59
                         This is useful for analyzing patterns based on time of day.
            - '7ds': Aggregates data from the past 7 days (one week) from the current date.
            - '14ds': Aggregates data from the past 14 days (two weeks) from the current date.

    """

    input_fn: str

    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Optional var
    col_suffix: str = ""
    groupby_cols = ["user"]

    def __post_init__(self) -> None:

        self.data = pd.read_parquet(self.input_fn)

        # Convert user to str
        self.data["user"] = self.data["user"].astype(str)

    def remove_timezone_info(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.tz_localize(None)
        # df["datetime"] = df["datetime"].dt.tz_localize(None)
        return df

    @progress_decorator
    def convert_copenhagen_time(self, df) -> pd.DataFrame():

        # Convert the 'datetime' column from milliseconds to a datetime object
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")

        # Set timezone to UTC (assuming the timestamps are in UTC)
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")

        # Convert to Copenhagen time (CET/CEST)
        df["datetime"] = df["datetime"].dt.tz_convert("Europe/Copenhagen")

        return df

    @progress_decorator
    def set_datetime_index(self, df) -> pd.DataFrame:

        # Set datatime
        df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
        # df['time'] = df['datetime']
        df.set_index(df["datetime"], inplace=True)
        df.index.name = None

        return df

    @progress_decorator
    def extract_features(self) -> pd.DataFrame:
        """
        Extract features based on the specified frequency.
            pd.DataFrame: A DataFrame containing the extracted features.

        Raises:
            NotImplementedError: This is a placeholder method and should be implemented in child classes.
        """
        raise NotImplementedError("This method should be implemented by child classes.")

    @progress_decorator
    def filter_outliers(self, df, cols):
        # Iterate over each specified column
        for col in cols:
            # Calculate the 95th percentile for the current column
            threshold = df[col].quantile(0.95)
            # Filter the DataFrame to keep only the data below the threshold
            df = df[df[col] < threshold]
        return df

    @progress_decorator
    def drop_duplicates_and_sort(self, data: pd.DataFrame) -> pd.DataFrame:
        data.sort_values(by=["user", "datetime"], inplace=True)
        # data = data.drop_duplicates(["user", "datetime"])

        return data

    @progress_decorator
    def remove_first_last_day(self, df):

        # Assert datetime index
        pd.api.types.is_datetime64_any_dtype(df.index)

        # Function to filter out the first and last day for each group
        def filter_days(group):
            # Determine the first and last day
            first_day = group.index.min().floor("D")
            last_day = group.index.max().floor("D")

            # Exclude rows from the first and last day
            return group[
                (group.index.floor("D") > first_day)
                & (group.index.floor("D") < last_day)
            ]

        # Group by 'user' and 'device' and apply the filter_days function
        return df.groupby(self.groupby_cols, group_keys=False).apply(filter_days)

    @progress_decorator
    # Roll over past n days and return summary of aggregated values
    def roll(self, df):
        # Sort by date first
        df = df.sort_values(["user", "date"])
        df.set_index("date", inplace=True)

        # Roll the dataframe based on frequency
        roll_map = {"14ds": 14, "7ds": 7, "3ds": 3}

        days = roll_map.get(self.frequency)
        if days:
            df = (
                df.groupby(self.groupby_cols)
                .rolling(days)
                .agg(["sum", "min", "max", "mean", "std"])
                .pipe(self.flatten_columns)
            )
        return df

    @progress_decorator
    def flatten_columns(self, df):
        """
        Flatten columns if they are 2-level
        """

        df.columns = [":".join(col).strip() for col in df.columns.values]

        return df

    @progress_decorator
    def rename_segment_columns(self, df):
        """
        Rename columns from time indicators to parts of the day.

        Parameters:
        - df: pandas.DataFrame with columns to rename.

        Returns:
        - DataFrame with renamed columns.
        """
        # Mapping of time indicators to parts of the day
        segments = {
            ":00": ":night",
            ":06": ":morning",
            ":12": ":afternoon",
            ":18": ":evening",
        }

        # Rename columns based on the mapping
        for time_indicator, segment in segments.items():
            df.columns = [col.replace(time_indicator, segment) for col in df.columns]

        # Append with sensor name
        df.columns = [f"{self.sensor_name}:{col}" for col in df.columns]

        df.columns = [f"{col}{self.col_suffix}" for col in df.columns]
        return df

    @progress_decorator
    def sum_segment(self, df, prefixes):
        for prefix in prefixes:
            cols = [col for col in df if col.startswith(prefix)]
            df[f"{prefix}:allday{self.col_suffix}"] = df[cols].sum(axis=1)
        return df

    @progress_decorator
    def normalize_within_user(self, df, prefixes):
        """
        For each user, create a normalize version of numerical columns
        """

        print(df.head())
        for prefix in prefixes:
            cols = [col for col in df if col.startswith(prefix)]
            for col in cols:
                df[f"{col}:within_norm"] = df.groupby(self.groupby_cols)[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )

        return df

    def normalize_between_user(self, df, prefixes):
        """
        For all users, create a normalize version of numerical columns
        """

        for prefix in prefixes:
            cols = [
                col
                for col in df
                if col.startswith(prefix) and not col.endswith("within_norm")
            ]
            for col in cols:
                df[f"{col}:between_norm"] = df[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )

        return df

    def normalize_segments(self, df, cols):
        """
        Normalizes specified segment columns within a DataFrame so that the sum of segments equals 1 for each row.

        Parameters:
        - df (pandas.DataFrame): The DataFrame containing the data to be normalized.
        - cols (list of str): A list of base column names to be normalized across specified segments.

        Returns:
        - pandas.DataFrame: The DataFrame with added normalized segment columns and sum columns for each of the base columns.
        """

        # Define the time segments to be normalized
        segments = [":night", ":morning", ":afternoon", ":evening"]

        # Loop through each base column specified in 'cols'
        for col in cols:
            # Column storing sum of segments
            sum_col = f"{col}:allday"

            # Ok, this code is dirty but I'll let it be
            if self.frequency != "4epochs":
                # Generate the full column names for each segment
                segment_cols = [
                    f"{col}{segment}:{self.frequency}:sum" for segment in segments
                ]

            else:
                segment_cols = [f"{col}{segment}" for segment in segments]

            segment_df = df[segment_cols].copy()

            # Calculate the sum of segment values for each row and store in the new column
            df[sum_col] = segment_df.sum(axis=1)

            # Generate column names for the normalized values
            segment_cols_norm = [f"{col}:proportion" for col in segment_cols]

            # Normalize each segment value by dividing by the sum and store in new normalized columns
            df[segment_cols_norm] = segment_df.div(df[sum_col], axis=0)

            # Replace resulting NaN values with 0
            df[segment_cols_norm] = df[segment_cols_norm].fillna(0)
        return df

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

    input_fns: [str]
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Optional var
    col_suffix: str = ""
    groupby_cols = ["user"]

    def __post_init__(self) -> None:

        res = []
        raw_files = self.input_fns.raw_files

        for fn in raw_files:
            df = pd.read_csv(fn, index_col=0)

            # Extract wave name from the file path if needed
            wave = fn.split("/")[-3]  # Assuming wave is in the second last directory

            # Rename
            df.rename(columns={"pid": "user"}, inplace=True)
            df["wave"] = wave
            res.append(df)

        # Output
        self.data = pd.concat(res)

        # Also, load pid mappings
        self.pid_mappings = pd.read_csv(self.input_fns.pid_mappings, index_col=0)

    def fill_nan_with_zeros(self, df, columns):
        df[columns] = df[columns].fillna(0)
        return df

    def re_id_returning_users(self, df):

        id_mappings = self.pid_mappings.copy()
        wave_cols = ["PID I", "PID II", "PID III", "PID IV"]
        new_wave_cols = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]

        # Retain only these columns
        id_mappings = id_mappings[wave_cols]

        # Rename waves
        cols_map = {
            "PID I": "INS-W_1",
            "PID II": "INS-W_2",
            "PID III": "INS-W_3",
            "PID IV": "INS-W_4",
        }
        id_mappings = id_mappings.rename(cols_map, axis="columns")

        # convert every numeric cell in the DataFrame to INS_W_XXX / INS_W_XXXX
        def map_to_sensor_id(val):
            """
            Map a numeric ID to the 'INS-W_' format:
            - If val < 1000 => 'INS-W_003' (zero-padded to three digits)
            - If val ≥ 1000 => 'INS-W_1320'
            Non-numeric or missing values are left unchanged.
            """
            if pd.isna(val):
                return np.nan
            try:
                n = int(val)
                return f"INS-W_{n:03d}" if n < 1000 else f"INS-W_{n}"
            except (ValueError, TypeError):
                return val

        # Apply to all columns
        id_mappings = id_mappings.apply(lambda col: col.map(map_to_sensor_id))

        # ID mapping function
        def build_uid(row):
            """Return unified ID based on how many waves the person appears in."""
            ids = [x for x in row[new_wave_cols] if pd.notna(x)]

            if len(ids) == 0:
                return np.nan  # no ID present in any wave
            if len(ids) == 1:
                return ids[0]  # single-wave subject => keep original code
            ids = sorted(ids)
            return f"{len(ids)}_{'-'.join(ids)}"  # e.g. 3_INS-W_001_INS_W_033_INS_W_192

        # Create a unique id
        id_mappings["unique_id"] = id_mappings.apply(build_uid, axis=1)

        # Melt
        id_mappings = pd.melt(
            id_mappings,
            id_vars=["unique_id"],
            value_vars=["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"],
            var_name="wave",
            value_name="user",
        )

        # Merge with original ids

        df = df.merge(id_mappings, on=["user", "wave"])

        # Replace user id with the new unique id
        df = df.drop(columns=["user"])
        df = df.rename(columns={"unique_id": "user"})

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
    def normalize_within_user(self, df, prefixes):
        """
        For each user, create a min-max normalized version of numerical col. mns
        """

        for prefix in prefixes:
            cols = [col for col in df if col.startswith(prefix)]
            for col in cols:
                df[f"{col}:within_norm"] = df.groupby(self.groupby_cols)[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )

        return df

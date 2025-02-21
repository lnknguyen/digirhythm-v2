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
        for fn in self.input_fns:
            df = pd.read_csv(fn, index_col=0)

            # Extract wave name from the file path if needed
            wave = fn.split("/")[-3]  # Assuming wave is in the second last directory

            # Rename
            df.rename(columns={"pid": "user"}, inplace=True)
            df["wave"] = wave
            res.append(df)

        # Output
        self.data = pd.concat(res)

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
        For each user, create a min-max normalized version of numerical columns
        """

        for prefix in prefixes:
            cols = [col for col in df if col.startswith(prefix)]
            for col in cols:
                df[f"{col}:within_norm"] = df.groupby(self.groupby_cols)[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )

        return df

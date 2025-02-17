from base import BaseProcessor
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import polars as pl

import os


@dataclass
class StepGarminProcessor(BaseProcessor):

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.sensor_name = "step"

    def clean(self, df) -> pd.DataFrame():

        # Replace negative values in column 'steps' with 0
        df.loc[df['steps'] < 0, 'steps'] = 0
        return df
        
    def rename(self, df) -> pd.DataFrame():

        df.rename(
            columns={
                "ParticipantID": "user",
                "local_time": "datetime",
                "hourly_steps": "steps",
            },
            inplace=True,
        )
        return df

    def resample(self, df, rule="6H") -> pd.DataFrame():

        df = df.groupby("user")["steps"].resample(rule).sum().reset_index()
        return df

    def extract_features(self) -> pd.DataFrame:

        df = (
            self.data.pipe(self.rename)
            .pipe(self.drop_duplicates_and_sort)
            .pipe(self.set_datetime_index)
            .pipe(self.remove_first_last_day)
            .pipe(self.clean)
            .pipe(self.resample)
            .pipe(self.pivot)
            .pipe(self.flatten_columns)
            .pipe(self.rename_segment_columns)
            .pipe(self.sum_segment, prefixes=["step"])
            .reset_index()
        )

        return df

    def pivot(self, df):
        """
        Pivot dataframe so that features are spread across columns
        Example: screen_use_00, screen_use_01, ..., screen_use_23
        """
        df.rename(
            columns={"level_0": "user", "level_2": "datetime"},
            inplace=True,
        )

        print(df.head())

        df["hour"] = pd.to_datetime(df["datetime"]).dt.strftime("%H")
        df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")

        # Pivot the table
        pivoted_df = df.pivot_table(
            index=["user", "date"],
            columns="hour",
            values=[
                "steps",
            ],
            fill_value=0,
        )

        return pivoted_df


def main():

    frequency = "4epochs"

    input_fn = snakemake.input[0]
    output_fn = snakemake.output[0]

    dfs = []
    processor = StepGarminProcessor(input_fn=input_fn)

    res = processor.extract_features().reset_index()
    res.to_csv(output_fn, index=False)


if __name__ == "__main__":
    main()

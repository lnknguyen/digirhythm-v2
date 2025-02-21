from base import BaseProcessor
from dataclasses import dataclass, field
import niimpy
import pandas as pd
import niimpy.preprocessing.screen as screen

import polars as pl


@dataclass
class ScreenProcessor(BaseProcessor):

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.sensor_name = "screen"
        self.frequency = "4epochs"

    def aware_adapter(self) -> pd.DataFrame:

        self.data["timestamp_index"] = self.data["datetime"]
        self.data.set_index("timestamp_index", inplace=True)
        self.data.index.name = None

    def extract_features(self) -> pd.DataFrame:
        prefixes = [
            "screen:screen_use_durationtotal",
            "screen:screen_use_count",
        ]

        # Agg daily events into 6H bins
        rule = "6H"

        wrapper_features = {
            screen.screen_duration: {
                "screen_column_name": "screen_status",
                "resample_args": {"rule": rule},
            },
            screen.screen_count: {
                "screen_column_name": "screen_status",
                "resample_args": {"rule": rule},
            },
        }

        empty_batt_data = pd.DataFrame()
        df = (
            self.data.pipe(self.drop_duplicates_and_sort)
            .pipe(self.remove_first_last_day)
            .pipe(
                screen.extract_features_screen,
                empty_batt_data,
                features=wrapper_features,
            )  # call niimpy to extract features with pre-defined time bin
            .reset_index()
            .pipe(self.filter_outliers, cols=["screen_use_durationtotal"])
            .pipe(self.pivot)
            .pipe(self.flatten_columns)
            .pipe(self.rename_segment_columns)
            .pipe(self.sum_segment, prefixes=prefixes)
            .reset_index()
            .pipe(self.roll)
            .pipe(
                self.normalize_within_user, prefixes=prefixes
            )  # normalize within-user features
            .pipe(
                self.normalize_between_user, prefixes=prefixes
            )  # normalize between-user features
            .pipe(self.normalize_segments, cols=prefixes)
        )

        return df

    def pivot(self, df):
        """
        Pivot dataframe so that features are spread across columns
        Example: screen_use_00, screen_use_01, ..., screen_use_23
        """

        df["user"] = df["level_0"]
        df["device"] = df["level_1"]
        df["datetime"] = df["level_2"]

        df["hour"] = pd.to_datetime(df["datetime"]).dt.strftime("%H")
        df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")

        # Pivot the table
        pivoted_df = df.pivot_table(
            index=["user", "date"],
            columns="hour",
            values=[
                "screen_use_durationtotal",
                "screen_use_count",
            ],
            fill_value=0,
        )

        return pivoted_df


def main():

    input_fn = snakemake.input[0]
    output_fn = snakemake.output[0]

    processor = ScreenProcessor(input_fn=input_fn)

    processor.aware_adapter()

    res = processor.extract_features().reset_index()
    res.to_csv(output_fn, index=False)


if __name__ == "__main__":
    main()

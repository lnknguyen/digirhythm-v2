from base import BaseProcessor
from dataclasses import dataclass
import niimpy
import niimpy.preprocessing.screen as screen
import pandas as pd
import os

path = os.path.abspath(niimpy.__file__)
print(path)

SCREEN_PREFIX = "screen:screen_on_durationtotal"
SCREEN_COL = "screen_on_durationtotal"
RESAMPLE_RULE = "6h"


@dataclass
class ScreenProcessor(BaseProcessor):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.sensor_name = "screen"
        self.frequency = "4epochs"

    def extract_features(self) -> pd.DataFrame:
        wrapper_features = {
            screen.screen_duration: {
                "screen_column_name": "screen_status",
                "resample_args": {"rule": RESAMPLE_RULE},
            }
        }

        return (
            self.data.pipe(self.convert_copenhagen_time)
            .pipe(self.set_datetime_index)
            .pipe(self.drop_duplicates_and_sort)
            .pipe(self.remove_first_last_day)
            .pipe(self.remove_timezone_info)
            .pipe(
                screen.extract_features_screen,
                pd.DataFrame(),
                features=wrapper_features,
            )
            .sort_index()
            .pipe(self.filter_outliers, cols=[SCREEN_COL])
            .reset_index()
            .pipe(self._pivot)
            .pipe(self.flatten_columns)
            .pipe(self.rename_segment_columns)
            .pipe(self.sum_segment, prefixes=[SCREEN_PREFIX])
            .reset_index()
            .pipe(self.roll)
            .pipe(self.normalize_within_user, prefixes=[SCREEN_PREFIX])
            .pipe(self.normalize_between_user, prefixes=[SCREEN_PREFIX])
            .pipe(self.normalize_segments, cols=[SCREEN_PREFIX])
        )

    def _pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(
            columns={"level_1": "user", "level_0": "device", "level_2": "datetime"}
        )
        df["hour"] = pd.to_datetime(df["datetime"]).dt.strftime("%H")
        df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
        return df.pivot_table(
            index=["user", "date"],
            columns="hour",
            values=[SCREEN_COL],
            fill_value=0,
        )


def main():
    processor = ScreenProcessor(input_fn=snakemake.input[0])

    res = (
        processor.extract_features()
        .reset_index()
        .rename(
            columns=lambda c: c.replace(
                "screen_on_durationtotal", "screen_use_durationtotal"
            )
        )
    )

    res.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
    main()

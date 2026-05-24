from base import BaseProcessor
from dataclasses import dataclass
import niimpy.preprocessing.screen as screen
import pandas as pd

SCREEN_PREFIXES = [
    "screen:screen_use_durationtotal",
    "screen:screen_use_count",
]
RESAMPLE_RULE = "6H"


@dataclass
class ScreenProcessor(BaseProcessor):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.sensor_name = "screen"
        self.frequency = "4epochs"

    def _setup_index(self) -> None:
        self.data["timestamp_index"] = self.data["datetime"]
        self.data.set_index("timestamp_index", inplace=True)
        self.data.index.name = None

    def extract_features(self) -> pd.DataFrame:
        wrapper_features = {
            screen.screen_duration: {
                "screen_column_name": "screen_status",
                "resample_args": {"rule": RESAMPLE_RULE},
            },
            screen.screen_count: {
                "screen_column_name": "screen_status",
                "resample_args": {"rule": RESAMPLE_RULE},
            },
        }

        return (
            self.data.pipe(self.drop_duplicates_and_sort)
            .pipe(self.remove_first_last_day)
            .pipe(
                screen.extract_features_screen,
                pd.DataFrame(),
                features=wrapper_features,
            )
            .reset_index()
            .pipe(self.filter_outliers, cols=["screen_use_durationtotal"])
            .pipe(self._pivot)
            .pipe(self.flatten_columns)
            .pipe(self.rename_segment_columns)
            .pipe(self.sum_segment, prefixes=SCREEN_PREFIXES)
            .reset_index()
            .pipe(self.roll)
            .pipe(self.normalize_within_user, prefixes=SCREEN_PREFIXES)
            .pipe(self.normalize_between_user, prefixes=SCREEN_PREFIXES)
            .pipe(self.normalize_segments, cols=SCREEN_PREFIXES)
        )

    def _pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            user=df["level_0"],
            device=df["level_1"],
            datetime=df["level_2"],
            hour=pd.to_datetime(df["level_2"]).dt.strftime("%H"),
            date=pd.to_datetime(df["level_2"]).dt.strftime("%Y-%m-%d"),
        )
        return df.pivot_table(
            index=["user", "date"],
            columns="hour",
            values=["screen_use_durationtotal", "screen_use_count"],
            fill_value=0,
        )


def main():
    processor = ScreenProcessor(input_fn=snakemake.input[0])
    processor._setup_index()
    processor.extract_features().reset_index().to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
    main()

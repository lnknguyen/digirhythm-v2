from base import BaseProcessor
from dataclasses import dataclass, field
import niimpy
import pandas as pd

import numpy as np
import polars as pl

import os

path = os.path.abspath(niimpy.__file__)
print(path)


@dataclass
class SleepProcessor(BaseProcessor):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

    def _filter_artifacts(self, df):
        df = df[(df.sleep_duration > 3) & (df.sleep_duration < 13)]
        return df

    def _process_sleep_data(self, df):

        for col in df.columns:
            print(col)

        """Process sleep data."""
        column_mapper = {
            "pid": "user",
            "f_slp:fitbit_sleep_summary_rapids_sumdurationasleepmain:allday": "sleep_duration",
            "f_slp:fitbit_sleep_summary_rapids_firstbedtimemain:allday": "sleep_start",
            "f_slp:fitbit_sleep_summary_rapids_firstwaketimemain:allday": "sleep_end",
        }

        df.rename(columns=column_mapper, inplace=True)
        df["sleep_duration"] /= 60
        df["sleep_onset"] = df["sleep_start"] / 60
        df["sleep_offset"] = (df["sleep_end"] / 60) + 24
        df["sleep_offset"] = df.groupby("user")["sleep_offset"].shift(-1)
        df["midsleep_hhmm"] = (df["sleep_offset"] - df["sleep_onset"]) / 2 + df[
            "sleep_onset"
        ]
        return df

    def extract_features(self) -> pd.DataFrame:

        df = (
            self.data.pipe(self._process_sleep_data)
            .pipe(self._filter_artifacts)
            .pipe(
                self.normalize_within_user, prefixes=["sleep_duration", "midsleep_hhmm"]
            )
        )

        # Retains only the following columns
        cols_to_retain = [
            "user",
            "date",
            "wave",
            "sleep_duration",
            "midsleep_hhmm",
            "sleep_onset",
            "sleep_offset",
            "sleep_duration:within_norm",
            "midsleep_hhmm:within_norm",
        ]
        df = df[cols_to_retain]

        return df


def main(input_fns, output_fn):

    processor = SleepProcessor(input_fns=input_fns)

    res = processor.extract_features().reset_index()
    res.to_csv(output_fn, index=False)


if __name__ == "__main__":

    main(snakemake.input, snakemake.output[0])

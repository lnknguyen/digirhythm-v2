from base import BaseProcessor
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import polars as pl

import os


@dataclass
class StepsProcessor(BaseProcessor):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

    def _process_step_data(self, df):
        column_mapper = {
            "pid": "user",
            "f_steps:fitbit_steps_intraday_rapids_sumsteps:night": "activity_night",
            "f_steps:fitbit_steps_intraday_rapids_sumsteps:morning": "activity_morning",
            "f_steps:fitbit_steps_intraday_rapids_sumsteps:afternoon": "activity_afternoon",
            "f_steps:fitbit_steps_intraday_rapids_sumsteps:evening": "activity_evening",
            "f_steps:fitbit_steps_intraday_rapids_sumsteps:allday": "activity_allday",
        }

        df.rename(columns=column_mapper, inplace=True)

        return df

    def extract_features(self) -> pd.DataFrame:

        df = self.data.pipe(self._process_step_data)

        # Retains only the following columns
        cols_to_retain = [
            "user",
            "date",
            "wave",
            "activity_night",
            "activity_morning",
            "activity_afternoon",
            "activity_evening",
            "activity_allday",
        ]
        df = df[cols_to_retain]

        df = self.fill_nan_with_zeros(
            df, columns="screen:screen_use_durationtotal:night"
        )
        return df


def main(input_fns, output_fn):

    processor = StepsProcessor(input_fns=input_fns)

    res = processor.extract_features().reset_index()
    res.to_csv(output_fn, index=False)


if __name__ == "__main__":

    main(snakemake.input, snakemake.output[0])

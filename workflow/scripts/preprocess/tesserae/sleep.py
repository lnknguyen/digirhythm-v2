from base import BaseProcessor
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import polars as pl

import os


@dataclass
class SleepGarminProcessor(BaseProcessor):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

    def rename(self, df) -> pd.DataFrame():

        df.rename(
            columns={"ParticipantID": "user", "local_time": "datetime", "adjusted_sleep_duration" : "sleep_duration"}, inplace=True
        )
        
        return df

    def extract_features(self) -> pd.DataFrame:

        df = self.data.pipe(self.rename)

        return df


def main():

    input_fn = snakemake.input[0]
    output_fn = snakemake.output[0]

    processor = SleepGarminProcessor(input_fn=input_fn)

    res = processor.extract_features().reset_index()
    res.to_csv(output_fn, index=False)


if __name__ == "__main__":
    main()

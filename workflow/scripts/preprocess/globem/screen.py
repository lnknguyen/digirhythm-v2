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

    def process_screen_data(self, df):
        """Process screen data."""
        column_mapper = {
            "pid": "user",
            "f_screen:phone_screen_rapids_sumdurationunlock:allday": "screen:screen_use_durationtotal:allday",
            "f_screen:phone_screen_rapids_sumdurationunlock:morning": "screen:screen_use_durationtotal:morning",
            "f_screen:phone_screen_rapids_sumdurationunlock:afternoon": "screen:screen_use_durationtotal:afternoon",
            "f_screen:phone_screen_rapids_sumdurationunlock:evening": "screen:screen_use_durationtotal:evening",
            "f_screen:phone_screen_rapids_sumdurationunlock:night": "screen:screen_use_durationtotal:night",
        }
        df.rename(columns=column_mapper, inplace=True)
        df["screen:screen_use_durationtotal:allday"] *= 60

        # Retains only the following columns
        cols_to_retain = [
            "user",
            "date",
            "wave",
            "screen:screen_use_durationtotal:morning",
            "screen:screen_use_durationtotal:afternoon",
            "screen:screen_use_durationtotal:evening",
            "screen:screen_use_durationtotal:night",
            "screen:screen_use_durationtotal:allday",
        ]
        df = df[cols_to_retain]

        return df

    def extract_features(self) -> pd.DataFrame:

        df = self.data.pipe(self.process_screen_data).pipe(
            self.normalize_within_user, prefixes=["screen:screen_use_durationtotal"]
        )

        return df


def main(input_fns, output_fn):

    processor = ScreenProcessor(input_fns=input_fns)

    res = processor.extract_features().reset_index()
    res.to_csv(output_fn, index=False)


if __name__ == "__main__":

    main(snakemake.input, snakemake.output[0])

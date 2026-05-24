from base import BaseProcessor
from dataclasses import dataclass
import pandas as pd

SCREEN_PREFIX = "screen:screen_use_durationtotal"

COLUMN_MAPPER = {
    "pid": "user",
    "f_screen:phone_screen_rapids_sumdurationunlock:allday": f"{SCREEN_PREFIX}:allday",
    "f_screen:phone_screen_rapids_sumdurationunlock:morning": f"{SCREEN_PREFIX}:morning",
    "f_screen:phone_screen_rapids_sumdurationunlock:afternoon": f"{SCREEN_PREFIX}:afternoon",
    "f_screen:phone_screen_rapids_sumdurationunlock:evening": f"{SCREEN_PREFIX}:evening",
    "f_screen:phone_screen_rapids_sumdurationunlock:night": f"{SCREEN_PREFIX}:night",
}

COLS_TO_RETAIN = [
    "user",
    "date",
    "wave",
    f"{SCREEN_PREFIX}:morning",
    f"{SCREEN_PREFIX}:afternoon",
    f"{SCREEN_PREFIX}:evening",
    f"{SCREEN_PREFIX}:night",
    f"{SCREEN_PREFIX}:allday",
]


@dataclass
class ScreenProcessor(BaseProcessor):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

    def _process_screen_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=COLUMN_MAPPER)[COLS_TO_RETAIN].copy()
        df[f"{SCREEN_PREFIX}:allday"] *= 60
        return df

    def extract_features(self) -> pd.DataFrame:
        return (
            self.data.pipe(self._process_screen_data)
            .pipe(self.normalize_within_user, prefixes=[SCREEN_PREFIX])
            .pipe(lambda df: df.fillna({f"{SCREEN_PREFIX}:night": 0}))
        )


def main(input_fns, output_fn):
    processor = ScreenProcessor(input_fns=input_fns)
    res = processor.re_id_returning_users(processor.extract_features().reset_index())
    res.to_csv(output_fn, index=False)


if __name__ == "__main__":
    main(snakemake.input, snakemake.output[0])

from base import BaseProcessor
from dataclasses import dataclass
import niimpy
import pandas as pd


@dataclass
class CallProcessor(BaseProcessor):

    def compute_total_call(self, df) -> pd.DataFrame:

        time_segments = ["morning", "afternoon", "evening", "night", "allday"]

        for segment in time_segments:
            df[f"call_total_count_{segment}"] = (
                df[f"call:outgoing_count:{segment}"]
                + df[f"call:incoming_count:{segment}"]
            )

            df[f"call_total_duration_{segment}"] = (
                df[f"call:outgoing_durationtotal:{segment}"]
                + df[f"call:incoming_durationtotal:{segment}"]
            )

        return df

    def extract_features(self) -> pd.DataFrame:

        df = self.data.copy()

        column_mapper = {
            "f_call:phone_calls_rapids_outgoing_sumduration:allday": "call:outgoing_durationtotal:allday",
            "f_call:phone_calls_rapids_outgoing_sumduration:morning": "call:outgoing_durationtotal:morning",
            "f_call:phone_calls_rapids_outgoing_sumduration:afternoon": "call:outgoing_durationtotal:afternoon",
            "f_call:phone_calls_rapids_outgoing_sumduration:evening": "call:outgoing_durationtotal:evening",
            "f_call:phone_calls_rapids_outgoing_sumduration:night": "call:outgoing_durationtotal:night",
            "f_call:phone_calls_rapids_incoming_sumduration:allday": "call:incoming_durationtotal:allday",
            "f_call:phone_calls_rapids_incoming_sumduration:morning": "call:incoming_durationtotal:morning",
            "f_call:phone_calls_rapids_incoming_sumduration:afternoon": "call:incoming_durationtotal:afternoon",
            "f_call:phone_calls_rapids_incoming_sumduration:evening": "call:incoming_durationtotal:evening",
            "f_call:phone_calls_rapids_incoming_sumduration:night": "call:incoming_durationtotal:night",
            "f_call:phone_calls_rapids_outgoing_count:allday": "call:outgoing_count:allday",
            "f_call:phone_calls_rapids_outgoing_count:morning": "call:outgoing_count:morning",
            "f_call:phone_calls_rapids_outgoing_count:afternoon": "call:outgoing_count:afternoon",
            "f_call:phone_calls_rapids_outgoing_count:evening": "call:outgoing_count:evening",
            "f_call:phone_calls_rapids_outgoing_count:night": "call:outgoing_count:night",
            "f_call:phone_calls_rapids_incoming_count:allday": "call:incoming_count:allday",
            "f_call:phone_calls_rapids_incoming_count:morning": "call:incoming_count:morning",
            "f_call:phone_calls_rapids_incoming_count:afternoon": "call:incoming_count:afternoon",
            "f_call:phone_calls_rapids_incoming_count:evening": "call:incoming_count:evening",
            "f_call:phone_calls_rapids_incoming_count:night": "call:incoming_count:night",
        }

        df.rename(columns=column_mapper, inplace=True)

        df = self.compute_total_call(df)

        # Retains only the following columns
        cols_to_retain = [
            "user",
            "date",
            "wave",
            "call_total_duration_allday",
            "call_total_duration_night",
            "call_total_duration_evening",
            "call_total_duration_afternoon",
            "call_total_duration_morning",
        ]

        # Add all mapped columns to the retention list
        mapped_columns = list(column_mapper.values())
        cols_to_retain.extend(mapped_columns)

        # Ensure unique columns in the list
        cols_to_retain = list(set(cols_to_retain))

        # Retain only the required columns
        df = df[cols_to_retain]

        return df


def main(input_fns, output_fn):

    processor = CallProcessor(input_fns=input_fns)

    res = processor.extract_features().reset_index()
    res = processor.re_id_returning_users(res)
    
    res.to_csv(output_fn, index=False)


if __name__ == "__main__":

    main(snakemake.input, snakemake.output[0])

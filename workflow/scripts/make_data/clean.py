import pandas as pd
import logging
from functools import reduce

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _filter(df, features):

    logging.info(f"Initial data shape before filtering: {df.shape}")

    # Report amount of NaN observations for each feature
    nan_counts = df[features].isna().sum()
    for feature, count in nan_counts.items():
        logging.info(f"Feature '{feature}' has {count} NaN observations")

    # Remove Nan features
    filtered_df = df.dropna(subset=features)
    logging.info(
        f"Data shape after removing NaNs based on features: {filtered_df.shape}"
    )

    # Remove days with zero activity (or step)
    if "activity_allday" in features:
        filtered_df = filtered_df[filtered_df["activity_allday"] != 0]
        logging.info(f"Data shape after removing zero activity: {filtered_df.shape}")

    if "screen_allday" in features:
        # Remove days with zero screen use
        filtered_df = filtered_df[filtered_df["screen_allday"] != 0]
        logging.info(f"Data shape after removing zero screen use: {filtered_df.shape}")

    if "sleep_duration" in features:
        # Remove days negative sleep duration
        filtered_df = filtered_df[filtered_df["sleep_duration"] > 0]
        logging.info(
            f"Data shape after removing negative sleep duration: {filtered_df.shape}"
        )

    return filtered_df


def _fill(df, features):

    call_features = [feature for feature in features if "call" in feature]
    logging.info(f"Features identified for filling missing call data: {call_features}")

    for feature in call_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)
            logging.info(f"Filled missing data with 0 for feature: {feature}")

    return df


def main(input_fns, output_fn):

    features = snakemake.params.features
    data = pd.read_csv(input_fns[0])

    # Assert that data contains column in features
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features in the data: {', '.join(missing_features)}")
    logging.info("All required features are present in the data.")

    # Fill
    df = _fill(data, features)

    # Filter
    df = _filter(df, features)

    # Export column names to txt
    colnames = df.columns.tolist()

    # Save to txt file
    with open("column_names.txt", "w") as file:
        for col in colnames:
            file.write(col + "\n")

    # Log final data shape before saving
    logging.info(f"Final data shape before saving to CSV: {df.shape}")

    df.to_csv(output_fn)


if __name__ == "__main__":
    main(snakemake.input, snakemake.output[0])

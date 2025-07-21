def re_id_returning_users(df, pid_mappings):

    id_mappings = pid_mappings.copy()
    wave_cols = ["PID I", "PID II", "PID III", "PID IV"]
    new_wave_cols = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]

    # Retain only these columns
    id_mappings = id_mappings[wave_cols]

    # Rename waves
    cols_map = {
        "PID I": "INS-W_1",
        "PID II": "INS-W_2",
        "PID III": "INS-W_3",
        "PID IV": "INS-W_4",
    }
    id_mappings = id_mappings.rename(cols_map, axis="columns")

    # convert every numeric cell in the DataFrame to INS_W_XXX / INS_W_XXXX
    def map_to_sensor_id(val):
        """
        Map a numeric ID to the 'INS-W_' format:
        - If val < 1000 => 'INS-W_003' (zero-padded to three digits)
        - If val ≥ 1000 => 'INS-W_1320'
        Non-numeric or missing values are left unchanged.
        """
        if pd.isna(val):
            return np.nan
        try:
            n = int(val)
            return f"INS-W_{n:03d}" if n < 1000 else f"INS-W_{n}"
        except (ValueError, TypeError):
            return val

    # Apply to all columns

    df["user"] = df["user"].apply(map_to_sensor_id)
    id_mappings = id_mappings.apply(lambda col: col.map(map_to_sensor_id))

    # ID mapping function
    def build_uid(row):
        """Return unified ID based on how many waves the person appears in."""
        ids = [x for x in row[new_wave_cols] if pd.notna(x)]

        if len(ids) == 0:
            return np.nan  # no ID present in any wave
        if len(ids) == 1:
            return ids[0]  # single-wave subject => keep original code
        ids = sorted(ids)
        return f"{len(ids)}_{'-'.join(ids)}"  # e.g. 3_INS-W_001_INS_W_033_INS_W_192

    # Create a unique id
    id_mappings["unique_id"] = id_mappings.apply(build_uid, axis=1)

    # Melt
    id_mappings = pd.melt(
        id_mappings,
        id_vars=["unique_id"],
        value_vars=["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"],
        var_name="wave",
        value_name="user",
    )

    # Merge with original ids

    df = df.merge(id_mappings, on=["user"])

    # Replace user id with the new unique id
    df = df.drop(columns=["user"])
    df = df.rename(columns={"unique_id": "user"})

    return df


ROOT = "/m/cs/work/luongn1"
# Load demographics
demo = []
for i in [2018, 2019, 2020, 2021]:
    df = pd.read_csv(f"{ROOT}/globem/demo_{i}.csv")
    demo.append(df)

# mapping
pid_mappings = pd.read_csv(f"{ROOT}/globem/pid_mappings.csv")

demo = pd.concat(demo)
demo = demo.rename(columns={"pid": "user"})
res = re_id_returning_users(demo, pid_mappings)
res.to_csv(f"{ROOT}/globem/demographics.csv", index=False)

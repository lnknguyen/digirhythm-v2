
# Define a list of studies and features
STUDIES = ["tesserae", "momo", "globem"]
features = ["screen", "sleep"]

# Clean up file folders
rule clean:
    
# Rule to concatenate feature files for a single study
rule concatenate_features:
    input:
        'data/interim/{study}/sleep_4epochs.csv',
        'data/interim/{study}/screen_4epochs.csv' 
    output:
        'data/processed/{study}/all_features.csv'
    shell:
        "python3 src/make_data/combine.py --input_fns {input} --output_fn {output}"


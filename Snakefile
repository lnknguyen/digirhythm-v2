STUDIES = ['momo']

configfile: 'config/latent.yaml'

rule all:
    input:
        expand('data/output/{study}/latent.csv', study=STUDIES)

rule latent:
    input: 
        'data/processed/{study}/all_features_clean.csv'
    output:
        'data/output/{study}/latent.csv'
    params:
        groupby = config['groupby'],
        features = config['features']
    script:
        'src/latent/make_latent.py'

# Clean up features by removing Nan and empty data
rule clean_features:
    input:
        'data/processed/{study}/all_features.csv'
    output:
        'data/processed/{study}/all_features_clean.csv'
    params:
        groupby = config['groupby'],
        features = config['features']
    script:
        "src/make_data/clean.py"


# Load in the files, rename the columns so that they match the schema
# then concat all the files
rule rename_and_concatenate:
    input:
        'data/interim/{study}/sleep_4epochs.csv',
        'data/interim/{study}/screen_4epochs.csv',
        'data/interim/{study}/steps_4epochs.csv',
        'data/interim/{study}/call_4epochs.csv'
    output:
        'data/processed/{study}/all_features.csv'
    script:
        "src/make_data/combine.py"

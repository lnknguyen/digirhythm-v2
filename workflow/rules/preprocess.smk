SENSORS = ['screen', 'call', 'steps', 'sleep']

# Write up report
rule write_report:
    input:
        expand('data/processed/{study}/all_features_clean.csv', study=['tesserae', 'globem'])
    output:
        report("report.png",  caption="../report/hello.rst", category="Feature statistics")
    conda:
        '../envs/python_env.yaml'
    script:
        "../scripts/make_data/report.py"

# Clean up features by removing Nan and empty data
rule clean_features:
    input:
        'data/processed/{study}/all_features.csv'
    output:
        'data/processed/{study}/all_features_clean.csv'
    conda:
        '../envs/python_env.yaml'
    params:
        groupby = config['groupby'],
        features = config['features']
    script:
        "../scripts/make_data/clean.py"

# Load in the files, rename the columns so that they match the schema
# then concat all the files
rule rename_and_concatenate:
    input:
        expand('data/interim/{{study}}/{sensor}_4epochs.csv', sensor=SENSORS)
    output:
        'data/processed/{study}/all_features.csv'
    conda:
        '../envs/python_env.yaml'
    params:
        groupby = config['groupby']
    script:
        "../scripts/make_data/combine.py"

# GLobem rule
WAVES = ['INS-W_1', 'INS-W_2', 'INS-W_3', 'INS-W_4']


rule extract_globem:
    input:
        expand('/m/cs/work/luongn1/globem/{wave}/FeatureData/{{sensor}}.csv', wave = WAVES)
    output:
        'data/interim/globem/{sensor}_4epochs.csv'
    conda:
        '../envs/python_env.yaml'
    script:
        '../scripts/preprocess/globem/{wildcards.sensor}.py'

# Tesserae rule
rule extract_tesserae:
    input:
        '/scratch/cs/tesserae/postprocessed_data/{sensor}.parquet'
    output:
        'data/interim/tesserae/{sensor}_4epochs.csv'
    conda:
        '../envs/python_env.yaml'
    script:
        '../scripts/preprocess/tesserae/{wildcards.sensor}.py'
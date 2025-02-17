import yaml



SENSORS = ['screen','calls', 'steps', 'sleep']
STUDIES = ['tesserae']


rule extract_tesserae:
    input:
        '/scratch/cs/tesserae/postprocessed_data/{sensor}.parquet'
    output:
        'data/interim/tesserae/{sensor}_4epochs.csv'
    conda:
        '../envs/python_env.yaml'
    script:
        '../scripts/preprocess/tesserae/{wildcards.sensor}.py'

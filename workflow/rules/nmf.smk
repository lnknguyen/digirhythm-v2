###########################################
#               NMF        #
###########################################
rule nmf:
    input:
        'data/processed/{study}/all_features_clean.csv'
    output:
        'out/{study}/nmf_components.csv'
    params:
        features = config['features']
    conda:
        "../envs/latent_env.yaml"
    script:
        '../scripts/nmf/nmf.py'

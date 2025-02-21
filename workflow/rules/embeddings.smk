

rule cluster_plot_embeddings:
    input:
        'data/output/{study}/embeddings.csv'
    output:
        report('data/output/{study}/cluster_embeddings.png')
    params:
        min_sample_size = config['hdbscan_parameters'][{study}]
    conda:
        "../envs/latent_env.yaml"
    script:
        '../scripts/latent/cluster_embeddings.py'

rule kmean_cluster:
    input: 
        'data/processed/{study}/all_features_clean.csv'
    output:
        'data/output/{study}/kmeans_cluster.csv'
    params:
        features = config['features']
    conda:
        "../envs/latent_env.yaml"
    script:
        '../scripts/latent/kmeans.py'

rule build_embeddings:
    input: 
        'data/processed/{study}/all_features_clean.csv'
    output:
        'data/output/{study}/embeddings.csv'
    conda:
        "../envs/latent_env.yaml"
    params:
        groupby = config['groupby'],
        features = config['features']
    resources:
        mem_mb=10000
    script:
        '../scripts/latent/make_embeddings.py'

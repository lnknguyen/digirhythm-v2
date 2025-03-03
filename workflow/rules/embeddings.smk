###########################################
#               CLUSTERING        #
###########################################

rule cluster:
    input: 
        'data/processed/{study}/all_features_clean.csv'
    output:
        clusters = 'out/{study}/{algo}_cluster.csv',
        centroids = 'out/{study}/{algo}_cluster_centroids.csv',
    params:
        features = config['features'],
        cluster_settings = lambda w: config["cluster_settings"]["{}".format(w.study)]
    conda:
        "../envs/latent_env.yaml"
    script:
        '../scripts/clusters/run.py'


###########################################
#      UMAP & CLUSTERING & PLOT   #
###########################################

rule build_embeddings:
    input: 
        'data/processed/{study}/all_features_clean.csv'
    output:
        'data/output/{study}/embeddings.csv'
    params:
        groupby = config['groupby'],
        features = config['features']
    resources:
        mem_mb=10000
    conda:
        "../envs/latent_env.yaml"
    script:
        '../scripts/embeddings/make_embeddings.py'

rule plot_embeddings:
    input:
        embeddings = 'data/output/{study}/embeddings.csv',
        labels = 'out/{study}/{algo}_cluster.csv',
    output:
        report('data/output/{study}/{algo}_cluster_embeddings.png')
    params:
        features = config['features'],
    conda:
        "../envs/latent_env.yaml"
    script:
        '../scripts/embeddings/plot_embeddings.py'

###########################################
#               CLUSTERING        #
###########################################
rule visualize_cluster:
    input:
        centroids = 'out/{study}/gmm_cluster_centroids.csv'
    output:
        centroids_viz = report('out/{study}/cluster_centroids.png'),
        distance_centroids_viz = report('out/{study}/distance_between_centroids.png'),
    params:
        features = config['features'],
    conda:
        "../envs/latent_env.yaml"
    script:
        '../scripts/clusters/visualize_cluster.py'

rule cluster:
    input: 
        'data/processed/{study}/all_features_clean.csv'
    output:
        clusters = 'out/clusters/{study}/{algo}_cluster.csv',
        centroids = 'out/clusters/{study}/{algo}_cluster_centroids.csv',
        covariances = 'out/clusters/{study}/{algo}_cluster_covariances.csv',
        scores = 'out/model_selection/{study}/{algo}_model_selection.csv',
    params:
        features = config['features'],
        cluster_settings = lambda w: config["cluster_settings"]["{}".format(w.study)]
    conda:
        "../envs/latent_env.yaml"
    script:
        '../scripts/clusters/run.py'

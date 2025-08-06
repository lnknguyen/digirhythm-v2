###########################################
#               CLUSTERING        #
###########################################

# Gather desired outputs for model selection, per-study
def model_selection_targets():
    outs = []
    for study, cs in config.get("cluster_settings", {}).items():
        if not cs.get("run_model_selection", False):
            continue
        algos = cs.get("algorithms")
        if algos is None:
            algos = [cs.get("algorithm")] if cs.get("algorithm") else []
        for algo in (algos if isinstance(algos, (list, tuple)) else [algos]):
            outs.append(f"out/model_selection/{study}/{algo}_model_selection.csv")
    return outs

rule run_model_selection:
    input: 
        'data/processed/{study}/all_features_clean.csv'
    output:
        scores = 'out/model_selection/{study}/{algo}_model_selection.csv'
    params:
        run_selection = True,
        features = lambda w: config["features"]["{}".format(w.study)],
        cluster_settings = lambda w: config["cluster_settings"]["{}".format(w.study)]
    conda:
        "../envs/latent_env.yaml"
    script:
        '../scripts/clusters/run.py'

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
        covariances = 'out/clusters/{study}/{algo}_cluster_covariances.csv'
    params:
        run_selection = False,
        features = lambda w: config["features"]["{}".format(w.study)],
        cluster_settings = lambda w: config["cluster_settings"]["{}".format(w.study)]
    conda:
        "../envs/latent_env.yaml"
    script:
        '../scripts/clusters/run.py'

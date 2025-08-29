rule signature:
    input:
        'out/clusters/{study}/{cluster_num}/gmm_cluster.csv'
    output:
        signature = 'out/signature/{study}/cluster_{cluster_num}/{window}/signature_{rank}_{dist}.csv',
        d_self = 'out/signature/{study}/cluster_{cluster_num}/{window}/signature_d_self_{rank}_{dist}.csv',
        d_ref = 'out/signature/{study}/cluster_{cluster_num}/{window}/signature_d_ref_{rank}_{dist}.csv'
    params:
        features = lambda w: config["features"]["{}".format(w.study)],
        splits = lambda w: config["signature"][w.study]["splits"],
        ranked = lambda w: w.rank,
        dist_method = lambda w: w.dist
    conda:
        "../envs/python_env.yaml"
    script:
        '../scripts/signature/signature.py'

rule transition_signature:
    input:
        'out/clusters/{study}/{cluster_num}/gmm_cluster.csv'
    output:
        signature = 'out/transition_signature/{study}/cluster_{cluster_num}/{window}/transition_signature_{dist}.csv',
        d_self = 'out/transition_signature/{study}/cluster_{cluster_num}/{window}/transition_signature_d_self_{dist}.csv',
        d_ref = 'out/transition_signature/{study}/cluster_{cluster_num}/{window}/transition_signature_d_ref_{dist}.csv'
    params:
        features = lambda w: config["features"]["{}".format(w.study)],
        splits = lambda w: config["signature"][w.study]["splits"],
        dist_method = lambda w: w.dist
    conda:
        "../envs/python_env.yaml"
    script:
        '../scripts/signature/transition_signature.py'


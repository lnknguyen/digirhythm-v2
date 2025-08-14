rule signature:
    input:
        'out/clusters/{study}/gmm_cluster.csv'
    output:
        signature = 'out/signature/{study}/signature_{rank}_{dist}.csv',
        d_self = 'out/signature/{study}/signature_d_self_{rank}_{dist}.csv',
        d_ref = 'out/signature/{study}/signature_d_ref_{rank}_{dist}.csv'
    params:
        features = lambda w: config["features"]["{}".format(w.study)],
        threshold_days = lambda w: config["signature"][w.study]["threshold_days"],
        splits = lambda w: config["signature"][w.study]["splits"],
        ranked = lambda w: w.rank,
        dist_method = lambda w: w.dist
    conda:
        "../envs/python_env.yaml"
    script:
        '../scripts/signature/signature.py'

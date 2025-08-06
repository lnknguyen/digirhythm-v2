
rule signature:
    input:
        'out/clusters/{study}/gmm_cluster.csv'
    output:
        signature = 'out/signature/{study}/signature_{rank}.csv',
        d_self = 'out/signature/{study}/signature_d_self_{rank}.csv',
        d_ref = 'out/signature/{study}/signature_d_ref_{rank}.csv'
    params:
        features = lambda w: config["features"]["{}".format(w.study)],
        ranked =lambda wc: wc.get("rank"),
        dist_method = "l2"
    conda:
        "../envs/python_env.yaml"
    script:
        '../scripts/signature.py'

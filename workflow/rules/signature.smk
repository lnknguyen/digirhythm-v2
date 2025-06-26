
rule signature:
    input:
        'out/{study}/gmm_cluster.csv'
    output:
        signature = 'out/{study}/signature_{rank}.csv',
        d_self = 'out/{study}/signature_d_self_{rank}.csv',
        d_ref = 'out/{study}/signature_d_ref_{rank}.csv'
    params:
        features = config['features'],
        ranked =lambda wc: wc.get("rank"),
        dist_method = "l2"
    conda:
        "../envs/python_env.yaml"
    script:
        '../scripts/signature.py'

# ---- Config defaults with clean overrides from `config` ----
STUDIES         = config.get("STUDIES",         ['tesserae', 'momo', 'globem'])
ALGOS           = config.get("ALGOS",           ['gmm'])
RANKS           = config.get("RANKS",           ['ranked', 'unranked'])
DIST_METHOD     = config.get("DIST_METHOD",     ['jsd', 'cosine'])
OPTIMAL_CLUSTER = config.get("OPTIMAL_CLUSTER", [6, 7, 8, 9, 10, 11])
#WINDOW          = config.get("WINDOW",          [60, 90, 180, 270])
WINDOW          = config.get("WINDOW",          [60])

def model_selection_targets():
    # keep your original implementation
    return []

def all_outputs():
    outs = []
    outs.extend(expand(
        'out/clusters/{study}/{cluster_num}/{algo}_cluster.csv',
        cluster_num=OPTIMAL_CLUSTER, study=STUDIES, algo=ALGOS
    ))
    outs.extend(expand(
        'out/clusters/{study}/{cluster_num}/{algo}_cluster_centroids.csv',
        cluster_num=OPTIMAL_CLUSTER, study=STUDIES, algo=ALGOS
    ))
    # signature
    outs.extend(expand(
        'out/signature/{study}/cluster_{cluster_num}/{window}/signature_{rank}_{dist}.csv',
        cluster_num=OPTIMAL_CLUSTER, window=WINDOW, study=STUDIES, rank=RANKS, dist=DIST_METHOD
    ))
    outs.extend(expand(
        'out/signature/{study}/cluster_{cluster_num}/{window}/signature_d_self_{rank}_{dist}.csv',
        cluster_num=OPTIMAL_CLUSTER, window=WINDOW, study=STUDIES, rank=RANKS, dist=DIST_METHOD
    ))
    outs.extend(expand(
        'out/signature/{study}/cluster_{cluster_num}/{window}/signature_d_ref_{rank}_{dist}.csv',
        cluster_num=OPTIMAL_CLUSTER, window=WINDOW, study=STUDIES, rank=RANKS, dist=DIST_METHOD
    ))
    # transition signature
    outs.extend(expand(
        'out/transition_signature/{study}/cluster_{cluster_num}/{window}/transition_signature_{dist}.csv',
        cluster_num=OPTIMAL_CLUSTER, window=WINDOW, study=STUDIES, dist=DIST_METHOD
    ))
    outs.extend(expand(
        'out/transition_signature/{study}/cluster_{cluster_num}/{window}/transition_signature_d_self_{dist}.csv',
        cluster_num=OPTIMAL_CLUSTER, window=WINDOW, study=STUDIES, dist=DIST_METHOD
    ))
    outs.extend(expand(
        'out/transition_signature/{study}/cluster_{cluster_num}/{window}/transition_signature_d_ref_{dist}.csv',
        cluster_num=OPTIMAL_CLUSTER, window=WINDOW, study=STUDIES, dist=DIST_METHOD
    ))

    outs.extend(model_selection_targets())
    # de-duplicate in case any lists overlap
    return sorted(set(outs))

rule clean_signature:
    message:
        "Cleaning: removing signature*.csv under out/signature/"
    shell:
        r"""
        test -d out/signature && find out/signature -type f -name 'signature*.csv' -delete || true
        """

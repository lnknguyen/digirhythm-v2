
STUDIES = ['tesserae', 'momo', 'globem', 'dtu']

ALGOS = ['gmm']

# ranked or unranked signature
RANKS = ["ranked", "unranked"]

# Different type of distance metric
DIST_METHOD = ["jsd", "cosine"]

def all_outputs():

    outputs = []
    outputs.extend(expand('out/clusters/{study}/{algo}_cluster.csv', study=STUDIES, algo=ALGOS))
    outputs.extend(expand('out/clusters/{study}/{algo}_cluster_centroids.csv', study=STUDIES, algo=ALGOS))

    # Signature, self and ref distance
    outputs.extend(expand('out/signature/{study}/signature_{rank}_{dist}.csv', study=STUDIES, rank=RANKS, dist=DIST_METHOD))
    outputs.extend(expand('out/signature/{study}/signature_d_self_{rank}_{dist}.csv', study=STUDIES, rank=RANKS, dist=DIST_METHOD))
    outputs.extend(expand('out/signature/{study}/signature_d_ref_{rank}_{dist}.csv', study=STUDIES, rank=RANKS, dist=DIST_METHOD))

    # Transition signature, self and ref distance
    outputs.extend(expand('out/transition_signature/{study}/transition_signature_{rank}_{dist}.csv', study=STUDIES, rank=RANKS, dist=DIST_METHOD))
    outputs.extend(expand('out/transition_signature/{study}/transition_signature_d_self_{rank}_{dist}.csv', study=STUDIES, rank=RANKS, dist=DIST_METHOD))
    outputs.extend(expand('out/transition_signature/{study}/transition_signature_d_ref_{rank}_{dist}.csv', study=STUDIES, rank=RANKS, dist=DIST_METHOD))

    # Optional: model selection
    outputs.extend(model_selection_targets())

    return outputs

rule clean_signature:
    message: "Cleaning: Removing signature.csv files"
    run:
        import os
        for study in STUDIES:
            path = f"out/{study}/signature.csv"
            if os.path.exists(path):
                os.remove(path)
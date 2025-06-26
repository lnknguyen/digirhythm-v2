
STUDIES = ['tesserae', 'momo', 'globem']
#STUDIES = ['tesserae', 'momo']
ALGOS = ['gmm']

# ranked or unranked signature
RANKS = ["ranked", "unranked"]

def all_outputs():

    outputs = []
    outputs.extend(expand('out/{study}/{algo}_cluster.csv', study=STUDIES, algo=ALGOS))
    outputs.extend(expand('out/{study}/{algo}_cluster_centroids.csv', study=STUDIES, algo=ALGOS))

    # Signature
    outputs.extend(expand('out/{study}/signature_{rank}.csv', study=STUDIES, rank=RANKS))

    return outputs

rule clean_signature:
    message: "Cleaning: Removing signature.csv files"
    run:
        import os
        for study in STUDIES:
            path = f"out/{study}/signature.csv"
            if os.path.exists(path):
                os.remove(path)

STUDIES = ['tesserae', 'momo', 'globem']
#STUDIES = ['globem']
ALGOS = ['gmm']

def all_outputs():

    outputs = []
    outputs.extend(expand('out/{study}/{algo}_cluster.csv', study=STUDIES, algo=ALGOS))
    outputs.extend(expand('out/{study}/{algo}_cluster_centroids.csv', study=STUDIES, algo=ALGOS))

    # NMF
    #outputs.extend(expand('out/{study}/nmf_components.csv', study=STUDIES))

    return outputs

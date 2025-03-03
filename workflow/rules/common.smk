
#STUDIES = ['tesserae', 'momo', 'globem']
STUDIES = ['globem']
ALGOS = ['gmm']

def all_outputs():
    
    outputs = expand('data/output/{study}/{algo}_cluster_embeddings.png', study=STUDIES, algo=ALGOS)
    outputs.extend(expand('out/{study}/{algo}_cluster.csv', study=STUDIES, algo=ALGOS))
    outputs.extend(expand('out/{study}/{algo}_cluster_centroids.csv', study=STUDIES, algo=ALGOS))
    outputs.append('report.png')
    return outputs

import pandas as pd

STUDIES = ['tesserae', 'momo', 'globem']

hdbscan_params =  (
    pd.read_csv(config["hdbscan_params"], sep="\t", dtype={"study": str})
    .set_index("study", drop=False)
    .sort_index()
)
def all_outputs():
    
    outputs = expand('data/output/{study}/cluster_embeddings.png', study=STUDIES)
    outputs.extend(expand('data/output/{study}/kmeans_cluster.csv', study=STUDIES))
    outputs.append('report.png')
    return outputs

def get_hdbscan_params(params):
    
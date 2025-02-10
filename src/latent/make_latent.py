import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def pca(df):
    
    print(df)
    return None

def main(input_fns, output_fns, params):

    # Load data
    data = pd.read_csv(input_fns[0])
    
    # Retrieve feature list
    groupby = params.groupby
    features_list = params.features
    cols = groupby + features_list
    
    # Keep only groupby and features columns
    df = data[groupby + features_list]

    pca_comp = pca(df)
    return 
    
if __name__ == "__main__":

    # Load features from the config file passed as an argument
    main(snakemake.input, snakemake.output[0], snakemake.params)
    print(snakemake.params.features)
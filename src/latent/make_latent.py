import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def pca(df, features):
    # Standardizing the features
    x = StandardScaler().fit_transform(df[features])

    # Creating PCA object and fitting data
    pca = PCA(n_components=2)  # Adjust n_components for your specific needs
    principal_components = pca.fit_transform(x)

    # Creating a DataFrame with principal components
    pca_df = pd.DataFrame(data=principal_components,
                          columns=['Principal Component 1', 'Principal Component 2'])

    return pca_df, pca

def plot_pca(pca_df, pca, output_path, title='PCA Plot'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df)
    plt.title(title)
    plt.xlabel('Principal Component 1 ({}%)'.format(round(pca.explained_variance_ratio_[0]*100, 2)))
    plt.ylabel('Principal Component 2 ({}%)'.format(round(pca.explained_variance_ratio_[1]*100, 2)))
    plt.grid(True)
    
    # Save plot to file
    plt.savefig(output_path)
    plt.close()

def main(input_fns, output_fns, params):
    # Load data
    data = pd.read_csv(input_fns[0])

    # Retrieve feature list and grouping
    groupby = params['groupby']
    features_list = params['features']
    cols = groupby + features_list

    # Keep only groupby and features columns
    df = data[cols]

    # Perform PCA
    pca_df, pca_model = pca(df, features_list)

    # Save PCA components to file
    pca_df.to_csv(output_fns[0], index=False)

    # Plot and save PCA
    plot_pca(pca_df, pca_model, 'pca.png')

if __name__ == "__main__":
    
    main(snakemake.input, snakemake.output, snakemake.params)

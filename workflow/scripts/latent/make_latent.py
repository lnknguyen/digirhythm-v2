import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

def find_optimal_components(data, max_components=8):

    scores = []
    for n in range(1, max_components + 1):

        
        print("running NMF")
        model = NMF(n_components=n, init='random', random_state=0)
        W = model.fit_transform(data)

        
        H = model.components_
        reconstructed = np.dot(W, H)
        
        # Calculate the cophenetic correlation coefficient
        Z = linkage(reconstructed, 'ward')
        cophenetic_corr, _ = cophenet(Z, pdist(reconstructed))

        scores.append(cophenetic_corr)

    # Plotting the cophenetic coefficients
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_components + 1), scores, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cophenetic Correlation Coefficient')
    plt.title('Cophenetic Coefficient per Number of Components')
    plt.grid(True)
    plt.show()

    return np.argmax(scores) + 1  # +1 as index is 0-based and components count is 1-based

def main(input_fns, output_fns, params):
    # Load data
    data = pd.read_csv(input_fns[0])

    # Retrieve feature list and grouping
    groupby = params['groupby']
    features = params['features']
    cols = groupby + features

    # Keep only groupby and features columns
    df = data[cols]

    print(features)

    # Check which columns have negative values
    negative_cols = [col for col in features if (df[col] < 0).any()]

    # Assert that there are no negative values in the specified columns
    assert not negative_cols, f"Negative values detected in columns: {', '.join(negative_cols)}"

    optimal_k = find_optimal_components(df[features])
    print("OTPIMAL K: ", optimal_k)
        
if __name__ == "__main__":
    
    main(snakemake.input, snakemake.output, snakemake.params)

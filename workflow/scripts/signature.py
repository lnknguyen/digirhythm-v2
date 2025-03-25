import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def split_chunk(df, window=None, id_col=None):
    return

def signature(df):
    return
    
def d_self(df):
    return

def d_ref(df):
    return

def main(input_fns, output_fns, params):

    # Load data
    data = pd.read_csv(input_fns[0])


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, snakemake.params)

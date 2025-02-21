import pandas as pd
import matplotlib.pyplot as plt

# Load DataFrame
input_file = snakemake.input[0]
output_file = snakemake.output[0]

df = pd.read_csv(input_file)

# Specify the column for the histogram
column_name = "activity_allday"

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(df[column_name], bins=30, edgecolor="black")
plt.title(f"Histogram of {column_name}")
plt.xlabel(column_name)
plt.ylabel("Frequency")

# Save the histogram
plt.savefig(output_file)
plt.close()

print(f"Histogram saved to {output_file}")

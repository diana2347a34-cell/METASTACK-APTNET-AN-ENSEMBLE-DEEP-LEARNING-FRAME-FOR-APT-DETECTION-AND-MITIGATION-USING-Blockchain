import pandas as pd
from scipy.stats import zscore

# Load the dataset
file_path = "/content/drive/MyDrive/Colab Notebooks/Diana/imputed_network_data1.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Display the dataset before normalization
print("Dataset Before Normalization:")
print(data.head())

# Select numeric columns for normalization
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Apply Z-score normalization
data[numeric_columns] = data[numeric_columns].apply(zscore)

# Display the dataset after normalization
print("\nDataset After Z-score Normalization:")
print(data.head())

# Save the normalized data back to a CSV file (optional)
output_file = "/content/drive/MyDrive/Colab Notebooks/Diana/zscore_normalized_data1.csv"
data.to_csv(output_file, index=False)

print(f"\nNormalized data saved to {output_file}")

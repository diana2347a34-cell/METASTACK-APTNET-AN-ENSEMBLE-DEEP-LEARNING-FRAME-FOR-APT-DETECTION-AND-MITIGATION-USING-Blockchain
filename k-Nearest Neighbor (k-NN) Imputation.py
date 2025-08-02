import pandas as pd
from sklearn.impute import KNNImputer

# Load the dataset
file_path = "/content/drive/MyDrive/Colab Notebooks/Diana/smoothed_network_data1.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Display dataset before imputation
print("Dataset with Missing Values:")
print(data.head())

# Initialize k-NN imputer
# n_neighbors defines the number of nearest neighbors
k = 5  # Adjust the value of k as needed
imputer = KNNImputer(n_neighbors=k)

# Apply imputation to numeric columns only
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Display dataset after imputation
print("\nDataset After k-NN Imputation:")
print(data.head())

# Save the imputed data back to a CSV file (optional)
output_file = "/content/drive/MyDrive/Colab Notebooks/Diana/imputed_network_data1.csv"
data.to_csv(output_file, index=False)

print(f"\nImputed data saved to {output_file}")

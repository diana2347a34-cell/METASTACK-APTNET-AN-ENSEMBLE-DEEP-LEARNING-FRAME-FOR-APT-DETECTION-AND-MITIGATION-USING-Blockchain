import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset from a CSV file
input_file = '/content/drive/MyDrive/Colab Notebooks/Diana/zscore_normalized_data1.csv'  # Path to the input CSV file
data = pd.read_csv(input_file)

# Step 2: Remove non-numeric columns (if any)
data_numeric = data.select_dtypes(include=['number'])

# Step 3: Handle missing values using SimpleImputer (e.g., fill with mean)
imputer = SimpleImputer(strategy='mean')  # You can choose 'mean', 'median', or 'most_frequent'
data_numeric_imputed = imputer.fit_transform(data_numeric)

# Step 4: Display original data before PCA
print("Original Data (Before PCA):")
print(data_numeric.head())  # Show the first few rows of the original dataset

# Step 5: Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric_imputed)

# Step 6: Apply PCA
n_components = 10  # You can change this based on the number of components you need
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(data_scaled)

# Step 7: Display PCA results (After PCA)
print("\nTransformed Data (After PCA):")
pca_columns = [f'PC{i+1}' for i in range(n_components)]
pca_df = pd.DataFrame(principal_components, columns=pca_columns)
print(pca_df.head())  # Show the first few rows of the transformed data

# Step 8: Save the PCA results to a new CSV file (without the original data)
output_file = '/content/drive/MyDrive/Colab Notebooks/Diana/pca_output_only.csv'  # Path to the output CSV file
pca_df.to_csv(output_file, index=False)

# Optionally, print the explained variance ratio
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset from a CSV file
input_file = '/content/drive/MyDrive/Colab Notebooks/Diana (1)/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv/zscore_normalized_data1.csv'  # Path to the input CSV file
data = pd.read_csv(input_file)

# Step 2: Remove non-numeric columns (if any)
data_numeric = data.select_dtypes(include=['number'])

# Step 3: Handle missing values using SimpleImputer (e.g., fill with mean)
imputer = SimpleImputer(strategy='mean')  # You can choose 'mean', 'median', or 'most_frequent'
data_numeric_imputed = imputer.fit_transform(data_numeric)

# Step 4: Display original data before PCA
print("Original Data (Before PCA):")
print(data_numeric.head())  # Show the first few rows of the original dataset

# Step 5: Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric_imputed)

# Step 6: Apply PCA
n_components = 10  # You can change this based on the number of components you need
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(data_scaled)

# Step 7: Display PCA results (After PCA)
print("\nTransformed Data (After PCA):")
pca_columns = [f'PC{i+1}' for i in range(n_components)]
pca_df = pd.DataFrame(principal_components, columns=pca_columns)
print(pca_df.head())  # Show the first few rows of the transformed data

# Step 8: Save the PCA results to a new CSV file (without the original data)
output_file = '/content/drive/MyDrive/Colab Notebooks/Diana (1)/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv/pca_output_only.csv'  # Path to the output CSV file
pca_df.to_csv(output_file, index=False)

# Optionally, print the explained variance ratio
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)import pandas as pd

# Load the data from the CSV file
data_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Diana (1)/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv/cleaned_dataset.csv')

# Extract the target column
target_column = data_df['label']

# Save the target column to a new CSV file
target_column.to_csv('/content/drive/MyDrive/Colab Notebooks/Diana (1)/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv/target_column1.csv', index=False)

# Display the target column
print("Target Column:")
print(target_column)
import pandas as pd

# Read the CSV files into DataFrames
df1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Diana (1)/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv/pca_output_only.csv')  # Replace with the path to your first CSV file
df2 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Diana (1)/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv/target_column1.csv')  # Replace with the path to your second CSV file
# Check the shapes of both DataFrames
print("Shape of DataFrame 1:", df1.shape)
print("Shape of DataFrame 2:", df2.shape)
# Ensure both DataFrames have the same number of rows
# Truncate the larger DataFrame to match the row count of the smaller one
min_rows = min(df1.shape[0], df2.shape[0])
df1 = df1.iloc[:min_rows]
df2 = df2.iloc[:min_rows]

# Concatenate DataFrames by columns
df_concatenated = pd.concat([df1, df2], axis=1)

# Display the shape of the concatenated DataFrame
print("Shape after concatenation:", df_concatenated.shape)

# Optionally, save the concatenated DataFrame to a new CSV file
df_concatenated.to_csv('/content/drive/MyDrive/Colab Notebooks/Diana (1)/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv/concatenated_output4.csv', index=False)
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Diana (1)/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv/concatenated_output4.csv')
df
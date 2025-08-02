import pandas as pd

# Load your dataset
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Diana/enp0s3-public-tuesday.pcap_Flow.csv')

# Display the number of missing values per column (optional)
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Remove rows with null values
df_cleaned = df.dropna()

# Save the cleaned dataset to a new CSV file
output_file = '/content/drive/MyDrive/Colab Notebooks/Diana/cleaned_dataset.csv'
df_cleaned.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to {output_file}.")
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Diana/cleaned_dataset.csv')
df
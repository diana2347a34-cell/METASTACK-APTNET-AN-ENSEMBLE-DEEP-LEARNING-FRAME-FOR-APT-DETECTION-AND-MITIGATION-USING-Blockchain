import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "/content/drive/MyDrive/Colab Notebooks/Diana/cleaned_dataset.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to inspect structure
print("Dataset Preview:")
print(data.head())

# Select the column to be smoothed (e.g., 'Flow Bytes/s')
# Ensure the column is numeric
column_to_smooth = 'Flow Bytes/s'  # Replace with the desired column name
if column_to_smooth not in data.columns:
    raise ValueError(f"Column '{column_to_smooth}' not found in the dataset.")

# Ensure no missing values in the selected column (fill or drop as needed)
data[column_to_smooth].fillna(data[column_to_smooth].mean(), inplace=True)

# Apply Savitzky-Golay filter
window_size = 51  # Ensure this is odd
poly_order = 3
smoothed_values = savgol_filter(data[column_to_smooth], window_size, poly_order)

# Add smoothed data to the DataFrame
data[f'{column_to_smooth}_smoothed'] = smoothed_values

# Save the smoothed data back to a CSV (optional)
output_file = "/content/drive/MyDrive/Colab Notebooks/Diana/smoothed_network_data1.csv"
data.to_csv(output_file, index=False)
# Plot the original and smoothed data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data[column_to_smooth], label='Original Data', color='blue', alpha=0.6)
plt.plot(data.index, data[f'{column_to_smooth}_smoothed'], label='Smoothed Data', color='red', linewidth=2)
plt.title(f"Savitzky-Golay Smoothing of '{column_to_smooth}'")
plt.xlabel('Index')
plt.ylabel(column_to_smooth)
plt.legend()
plt.grid(True)
plt.show()

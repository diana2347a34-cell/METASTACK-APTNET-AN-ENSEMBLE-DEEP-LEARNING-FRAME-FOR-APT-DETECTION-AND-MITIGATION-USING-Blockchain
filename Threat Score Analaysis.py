import pandas as pd
import numpy as np
import skfuzzy as fuzz

# Load data from CSV
data_path = '/content/drive/MyDrive/Colab Notebooks/Diana/concatenated_output2.csv'
df = pd.read_csv(data_path)

# Example fuzzy sets
impact = np.arange(0, 11, 1)
exploit = np.arange(0, 11, 1)
threat = np.arange(0, 101, 1)

# Define fuzzy membership functions
impact_low = fuzz.trimf(impact, [0, 0, 5])
impact_medium = fuzz.trimf(impact, [0, 5, 10])
impact_high = fuzz.trimf(impact, [5, 10, 10])

exploit_low = fuzz.trimf(exploit, [0, 0, 5])
exploit_medium = fuzz.trimf(exploit, [0, 5, 10])
exploit_high = fuzz.trimf(exploit, [5, 10, 10])

threat_low = fuzz.trimf(threat, [0, 0, 50])
threat_medium = fuzz.trimf(threat, [0, 50, 100])
threat_high = fuzz.trimf(threat, [50, 100, 100])

# Example fuzzy rule base
def fuzzy_rulebase(impact_level, exploit_level):
    impact_level_low = fuzz.interp_membership(impact, impact_low, impact_level)
    impact_level_medium = fuzz.interp_membership(impact, impact_medium, impact_level)
    impact_level_high = fuzz.interp_membership(impact, impact_high, impact_level)

    exploit_level_low = fuzz.interp_membership(exploit, exploit_low, exploit_level)
    exploit_level_medium = fuzz.interp_membership(exploit, exploit_medium, exploit_level)
    exploit_level_high = fuzz.interp_membership(exploit, exploit_high, exploit_level)

    # Apply fuzzy rules
    rule1 = np.fmin(impact_level_low, exploit_level_low)
    rule2 = np.fmin(impact_level_medium, exploit_level_medium)
    rule3 = np.fmin(impact_level_high, exploit_level_high)

    return rule1, rule2, rule3

# Compute threat scores using fuzzy inference
def compute_threat_score(impact_level, exploit_level):
    # Clip the input levels to the fuzzy set range (0-10)
    impact_level = np.clip(impact_level, 0, 10)
    exploit_level = np.clip(exploit_level, 0, 10)

    # Ensure that the input levels are within the expected range
    if impact_level < 0 or impact_level > 10 or exploit_level < 0 or exploit_level > 10:
        return np.nan  # Return NaN if input values are outside expected range

    rule1, rule2, rule3 = fuzzy_rulebase(impact_level, exploit_level)

    # Apply fuzzy set membership to aggregate the rules
    threat_activation_low = np.fmin(rule1, threat_low)
    threat_activation_medium = np.fmin(rule2, threat_medium)
    threat_activation_high = np.fmin(rule3, threat_high)

    # Aggregate the results
    aggregated = np.fmax(threat_activation_low,
                         np.fmax(threat_activation_medium, threat_activation_high))

    # Perform defuzzification (centroid method)
    if np.sum(aggregated) == 0:  # If no membership values, return NaN
        return np.nan
    threat_score = fuzz.defuzz(threat, aggregated, 'centroid')
    return threat_score

# Create a list to store the threat scores and corresponding activities
results = []

# Iterate over each row in the DataFrame and compute threat scores
for index, row in df.iterrows():
    impact_level = row['PC4']
    exploit_level = row['PC1']
    threat_score = compute_threat_score(impact_level, exploit_level)

    # Check for NaN results (in case no valid threat score was computed)
    if not np.isnan(threat_score):
        results.append([row['Activity'], threat_score])

# Convert the results into a DataFrame
results_df = pd.DataFrame(results, columns=['Activity', 'Threat_Score'])

# Save the DataFrame to a CSV file
output_path = '/content/drive/MyDrive/Colab Notebooks/Diana/threat_scores.csv'
results_df.to_csv(output_path, index=False)

print(f"Threat scores have been saved to {output_path}")
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Diana/threat_scores.csv')
df
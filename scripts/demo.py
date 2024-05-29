import pandas as pd
from Levenshtein import distance as levenshtein_distance
from itertools import chain
import numpy as np
from pyod.models.iforest import IForest

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Read the dataset and truth file
df = pd.read_csv('./tracker-files/test2.csv')
truth_df = pd.read_csv('./tracker-files/test2_tracker.csv')

# Specify the column to analyze
column_to_process = 'Primary Fur Color'

# Find indices of missing and blank values
missing_indices = df[df[column_to_process].isna()].index.tolist()
#print(f"Missing values count: {len(missing_indices)}, indices: {missing_indices}")

# Compute the frequency of each unique value
counts = df[column_to_process].value_counts()

# Total counts and proportions
total_counts = counts.sum()
proportions = counts / total_counts

# Set thresholds
threshold = 0.01
anomaly_threshold = 0.8
misspelling_threshold = 0.3

# Classify values based on their proportions
low_proportion_values = proportions[proportions < threshold].index
high_proportion_values = proportions[proportions >= threshold].index

# Initialize data structures
misspellings = {}
anomalies = {}
misspelling_indices = {}
anomaly_indices = {}

def compute_distances(low_val, high_proportion_values):
    distances = []
    min_distance = float('inf')
    for high_val in high_proportion_values:
        max_len = max(len(low_val), len(high_val))
        edit_dist = levenshtein_distance(low_val, high_val)
        normalized_dist = edit_dist / max_len
        if normalized_dist < min_distance:
            min_distance = normalized_dist
        distances.append(normalized_dist)
    return distances, min_distance

def analyze_values(low_proportion_values, high_proportion_values, calculate_misspellings=False):
    for low_val in low_proportion_values:
        distances, min_distance = compute_distances(low_val, high_proportion_values)
        if calculate_misspellings and min_distance <= misspelling_threshold:
            misspellings[low_val] = min_distance
            misspelling_indices[low_val] = df[df[column_to_process] == low_val].index.tolist()
        average_distance = sum(distances) / len(high_proportion_values)
        if average_distance > anomaly_threshold:
            anomalies[low_val] = average_distance
            anomaly_indices[low_val] = df[df[column_to_process] == low_val].index.tolist()

# Calculate misspellings optionally
calculate_misspellings = True # Change this to False to skip misspelling calculation
analyze_values(low_proportion_values, high_proportion_values, calculate_misspellings)

#if(calculate_misspellings):
#    print("\nMisspellings (min average distance <= {}):".format(misspelling_threshold))
#    for misspelling, min_avg_dist in misspellings.items():
#        print(f"{misspelling}: {min_avg_dist} at indices {misspelling_indices[misspelling]}")

#print("\nAnomalies (average distance > {}):".format(anomaly_threshold))
#for anomaly, avg_dist in anomalies.items():
#    print(f"{anomaly}: {avg_dist} at indices {anomaly_indices[anomaly]}")

# PyOD for numerical outliers detection
numerical_columns = df.select_dtypes(include=numerics).columns

# Train Isolation Forest model for each numerical column
outliers = {}
for column in numerical_columns:
    numerical_data = df[column].values.reshape(-1, 1)
    if_model = IForest(contamination=0.1, random_state=42)
    if_model.fit(numerical_data)
    outliers[column] = if_model.predict(numerical_data)

# Get outlier indices for each numerical column
outlier_indices = {column: np.where(outliers[column] == 1)[0] for column in numerical_columns}

# Print outlier indices for each numerical column
for column in numerical_columns:
    print(f"Numerical Outlier Indices for {column}: {outlier_indices[column]}")



# Calculate TP
def is_accurate(detected_indices, truth_df, truth_type):
    if(truth_type == 4):
        return sum(truth_df.loc[detected_indices, column_to_process] == truth_type)
    else:
        flattened_indices = list(chain.from_iterable(detected_indices))
        return sum(truth_df.loc[flattened_indices, column_to_process] == truth_type)

# Calculate Recall
def calculate_recall(detected_indices, truth_df, truth_type):
    TP = is_accurate(detected_indices, truth_df, truth_type)
    FN = sum(truth_df[column_to_process] == truth_type) - TP
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return recall

accurate_misspellings = is_accurate(misspelling_indices.values(), truth_df, 2)
accurate_anomalies = is_accurate(anomaly_indices.values(), truth_df, 3)
accurate_missing = is_accurate(missing_indices, truth_df, 4)

total_misspellings_detected = sum(len(indices) for indices in misspelling_indices.values())
total_anomalies_detected = sum(len(indices) for indices in anomaly_indices.values())
total_missing_detected = len(missing_indices)

precision_misspellings = (accurate_misspellings / total_misspellings_detected) if total_misspellings_detected > 0 else 0
precision_anomalies = (accurate_anomalies / total_anomalies_detected) if total_anomalies_detected > 0 else 0
precision_missing = (accurate_missing / total_missing_detected) if total_missing_detected > 0 else 0
print(f"Misspelling precision: {precision_misspellings:.2%}")
print(f"Anomaly precision: {precision_anomalies:.2%}")
print(f"Missing value precision: {precision_missing:.2%}")

recall_misspellings = calculate_recall(misspelling_indices.values(), truth_df, 2)
recall_anomalies = calculate_recall(anomaly_indices.values(), truth_df, 3)
recall_missing = calculate_recall(missing_indices, truth_df, 4)
print(f"Misspelling recall: {recall_misspellings:.2%}")
print(f"Anomaly recall: {recall_anomalies:.2%}")
print(f"Missing value recall: {recall_missing:.2%}")

total_TP = accurate_misspellings + accurate_anomalies + accurate_missing
total_FN = (sum(truth_df[column_to_process] == 2) - accurate_misspellings) + \
           (sum(truth_df[column_to_process] == 3) - accurate_anomalies) + \
           (sum(truth_df[column_to_process] == 4) - accurate_missing)
total_detections = (
    sum(len(indices) for indices in misspelling_indices.values()) +
    sum(len(indices) for indices in anomaly_indices.values()) +
    len(missing_indices)
)

total_precision = total_TP / total_detections if total_detections > 0 else 0
total_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0

print(f"Number of TP in misspelling detections: {accurate_misspellings}")
print(f"Number of TP in anomaly detections: {accurate_anomalies}")
print(f"Number of TP in missing value detections: {accurate_missing}")
print(f"Total number of detections: {total_detections}")
print(f"Overall precision: {total_precision:.2%}")
print(f"Overall recall: {total_recall:.2%}")


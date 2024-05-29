import pandas as pd
import numpy as np
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF

def evaluate_model(model, df, tracker, col, metrics, dataset_name):
    # Check if column is contaminated
    if tracker[col].eq(1).all():
        print(f"Column {col} has not been contaminated, skipping detection.")
        return

    # Define indices where data is not NaN and not marked as outlier in tracker
    valid_indices = (df[col].notna()) & (tracker[col] != 4)

    # Define indices specifically for those marked as outliers
    outlier_indices = (df[col].isna()) & (tracker[col] == 5)

    # Filter data for valid entries
    valid_data = df.loc[valid_indices, col].to_frame()

    # Fit the model on non-null and non-outlier data
    if not valid_data.empty:
        model.fit(valid_data)

    # Initialize predictions with zeros
    predictions = np.zeros(df.shape[0], dtype=int)

    # Perform predictions where data is valid
    if not valid_data.empty:
        predictions[valid_indices] = model.predict(valid_data)

    # Manually adjust predictions for known outliers based on tracker
    predictions[outlier_indices] = 1

    # Tracker comparison only for valid data and known outliers
    actual_outliers = tracker[col] == 5
    predicted_outliers = predictions == 1

    # Calculate true positives
    true_positives = np.sum(predicted_outliers & actual_outliers)

    # Calculate precision and recall
    precision = true_positives / np.sum(predicted_outliers) if np.sum(predicted_outliers) > 0 else 0
    recall = true_positives / np.sum(actual_outliers) if np.sum(actual_outliers) > 0 else 0

    # Update metrics
    metrics[dataset_name][model.__class__.__name__]['precision'].append(precision)
    metrics[dataset_name][model.__class__.__name__]['recall'].append(recall)

    # Output results
    print(f"{model.__class__.__name__} on column {col}: Precision = {precision:.2f}, Recall = {recall:.2f}")

# List of models to apply
models = [KNN(), IForest(), HBOS(), LOF()]

# Initialize metrics dictionary
metrics = {}

# Load and process each dataset
for i in range(1, 5):  # Assuming there are 4 datasets from test_num1 to test_num4
    dataset_path = f'test_num{i}.csv'
    tracker_path = f'test_num{i}_tracker.csv'
    dataset_name = f'test_num{i}'

    print(f"Loading datasets: {dataset_path} and corresponding tracker")
    dataset = pd.read_csv(dataset_path)
    tracker = pd.read_csv(tracker_path)

    # Initialize dataset specific metrics
    metrics[dataset_name] = {model.__class__.__name__: {'precision': [], 'recall': []} for model in models}

    for column in dataset.columns:
        if dataset[column].dtype == np.int64 or dataset[column].dtype == np.float64:
            print(f"Processing column {column}...")
            for model in models:
                evaluate_model(model, dataset, tracker, column, metrics, dataset_name)

# Calculate and print average precision and recall for each model per dataset
for dataset_name, dataset_metrics in metrics.items():
    print(f"\nMetrics for {dataset_name}:")
    for model_name, values in dataset_metrics.items():
        avg_precision = np.mean(values['precision'])
        avg_recall = np.mean(values['recall'])
        print(f"{model_name}: Average Precision = {avg_precision:.2f}, Average Recall = {avg_recall:.2f}")

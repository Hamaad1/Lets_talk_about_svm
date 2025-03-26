import os
import time
import json
import numpy as np
import pandas as pd
from itertools import product

#progress_file = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/progress_file.json"

################################this code is new and more optimize then the below code replace below code with this########################33
def save_progress(new_entry, progress_file=None):
    """Append a new processed configuration to the progress file efficiently."""
    if progress_file is None:
        progress_file = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/progress_file.jsonl"

    # Use a set for fast lookup of processed entries
    processed_entries = load_progress(progress_file, as_set=True)

    # Check if the new entry already exists in the progress
    new_entry_str = json.dumps(new_entry)  # Convert to a string for easy comparison
    if new_entry_str not in processed_entries:
        with open(progress_file, 'a') as file:
            file.write(new_entry_str + '\n')  # Append as a new line

def load_progress(progress_file=None, as_set=False):
    """Load all previously processed configurations from the progress file."""
    if progress_file is None:
        progress_file = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/progress_file.jsonl"

    if not os.path.exists(progress_file):
        return set() if as_set else []

    with open(progress_file, 'r') as file:
        processed = [json.loads(line.strip()) for line in file]

    return set(json.dumps(entry) for entry in processed) if as_set else processed

def save_progress_old(new_entry, progress_file=None):#='progress_file.json'):
    """Append a new processed configuration to the progress file."""
    if progress_file is None:
        progress_file = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/progress_file.json"
    
    # Check if the progress file exists
    if os.path.exists(progress_file):
        # Load existing progress
        with open(progress_file, 'r') as file:
            progress = json.load(file)
    else:
        progress = []

    # Add the new entry if it's not already in the list
    if new_entry not in progress:
        progress.append(new_entry)

    # Save the updated progress back to the file
    with open(progress_file, 'w') as file:
        json.dump(progress, file)
        file.flush() 


def load_progress_old(progress_file=None):#='progress_file.json'):
    """Load all previously processed configurations from the progress file."""
    if progress_file is None:
        progress_file = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/progress_file.json"

    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            return json.load(file)  # Return the list of processed configurations
    return []  # Return an empty list if no progress file exists


def files_exist(folder, rep):
    """
    Checks if the individual error and prediction files already exist for the given repetition in the specified folder.
    
    Parameters:
    - folder: The directory where the results are stored.
    - rep: The repetition number (used for naming the files).
    
    Returns:
    - (bool): True if both the individual error and prediction files exist, False otherwise.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    # Define file names based on repetition
    individual_errors_file = os.path.join(folder, f"error_rep{rep+1}.csv")
    predictions_file = os.path.join(folder, f"predictions_rep{rep+1}.csv")
    
    # Check if both files exist
    return os.path.exists(individual_errors_file) and os.path.exists(predictions_file)

    
def prepare_data(dataset_params, processed_data):
    """
    Prepares the data for training and testing by extracting features and labels.

    Parameters:
    - dataset_params: A dictionary containing dataset-specific parameters.
    - processed_data: A dictionary with processed data containing RSSI and coordinates.

    Returns:
    - X: Features for training.
    - y: Labels (Latitude, Longitude, Altitude) for training.
    - X_testing: Features for testing.
    - y_testing: Labels (Latitude, Longitude, Altitude) for testing.
    """
    # Fetching non-detected values from dataset_params
    minValueDetected = dataset_params.get('minValueDetected', 0)
    defaultNonDetectedValue = dataset_params.get('defaultNonDetectedValue', 100)
    newNonDetectedValue = dataset_params.get('newNonDetectedValue', 0)

    rsamples1 = dataset_params.get('rsamples1')#, None
    osamples1 = dataset_params.get('osamples1')#, None
    nmacs1 = dataset_params.get('nmacs1')#, None
    rsamples = dataset_params.get('rsamples')
    osamples = dataset_params.get('osamples')
    nmacs = dataset_params.get('nmacs')

    # Extract the training and testing data
    train_rssi = pd.DataFrame(processed_data['trnrss'])
    test_rssi = pd.DataFrame(processed_data['tstrss'])
    train_coords = pd.DataFrame(
        processed_data['trncrd'], columns=['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']
    )
    test_coords = pd.DataFrame(
        processed_data['tstcrd'], columns=['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']
    )
    
    # Validate shapes for concatenation
    if train_rssi.shape[0] != train_coords.shape[0]:
        raise ValueError("Mismatch in number of training samples between RSSI and coordinates.")
    if test_rssi.shape[0] != test_coords.shape[0]:
        raise ValueError("Mismatch in number of testing samples between RSSI and coordinates.")


    # Concatenate the DataFrames
    train_df_combined = pd.concat([train_coords[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']], train_rssi], axis=1)
    X = train_df_combined.iloc[:, 5:]  # Features for training
    y = train_df_combined[['Latitude', 'Longitude', 'Altitude']].values  # Labels for training

    test_df_combined = pd.concat([test_coords[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']], test_rssi], axis=1)
    X_testing = test_df_combined.iloc[:, 5:]  # Features for testing
    y_testing = test_df_combined[['Latitude', 'Longitude', 'Altitude']].values  # Labels for testing

    return X, y, X_testing, y_testing, rsamples1, osamples1, nmacs1, rsamples, osamples, nmacs, minValueDetected, defaultNonDetectedValue, newNonDetectedValue


def fit_and_evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val, X_testing_scaled, y_testing):
    """
    Fits the model, makes predictions, and calculates errors.

    Parameters:
    - model: The machine learning model to be trained.
    - X_train_scaled: Scaled training features.
    - y_train: Training targets.
    - X_val_scaled: Scaled validation features.
    - y_val: Validation targets.
    - X_testing_scaled: Scaled testing features.
    - y_testing: Testing targets.

    Returns:
    - fit_time: Time taken to fit the model.
    - pred_time: Time taken to make predictions.
    - mean_3d_error_val: Mean 3D error for the validation set.
    - mean_3d_error_testing: Mean 3D error for the testing set.
    - errors_testing: List of individual errors for the testing set.
    - y_pred_testing: Predicted values for the testing set.
    """
    # Fit the model
    start_time_fit = time.time()
    model.fit(X_train_scaled, y_train)
    end_time_fit = time.time()
    fit_time = end_time_fit - start_time_fit

    # Validate the model
    y_pred_val = model.predict(X_val_scaled)
    errors_val = np.linalg.norm(y_val - y_pred_val, axis=1)
    mean_3d_error_val = np.round(np.mean(errors_val), 5)

    # Test the model
    start_time_pred = time.time()
    y_pred_testing = model.predict(X_testing_scaled)
    end_time_pred = time.time()
    pred_time = end_time_pred - start_time_pred

    errors_testing = np.linalg.norm(y_testing - y_pred_testing, axis=1)
    mean_3d_error_testing = np.round(np.mean(errors_testing), 5)

    return fit_time, pred_time, mean_3d_error_val, mean_3d_error_testing, errors_testing, y_pred_testing

def save_results(folder, rep, errors_testing, y_pred_testing):
    """
    Creates directories and saves errors and predictions to CSV files.

    Parameters:
    - folder: The directory where the results should be saved.
    - rep: The current repetition number (used for naming files).
    - errors_testing: Array of individual errors for the testing set.
    - y_pred_testing: Predicted values for the testing set.

    Returns:
    - individual_errors_file: Path to the saved individual errors CSV file.
    - predictions_file: Path to the saved predictions CSV file.
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the individual errors to a CSV file
    errors_df = pd.DataFrame({'Error': np.round(errors_testing, 5)})
    individual_errors_file = os.path.join(folder, f"error_rep{rep+1}.csv")
    errors_df.to_csv(individual_errors_file, index=False)

    # Save the predictions to a CSV file
    predictions_df = pd.DataFrame(y_pred_testing, columns=['Longitude', 'Latitude', 'Altitude'])
    predictions_file = os.path.join(folder, f"predictions_rep{rep+1}.csv")
    predictions_df.to_csv(predictions_file, index=False)

    return individual_errors_file, predictions_file


def generate_param_combinations(config):
    param_combinations = []
    for kernel_value in config['kernel']:
        if kernel_value == 'linear':
            param_combinations.extend(product(
                config['train_test_splits'],
                [kernel_value],
                config['C'],
                config['epsilon'],
                config['tol'],
                [None],  # Gamma not used
                [None],  # Degree not used
                [None]   # Coef0 not used
            ))
        elif kernel_value == 'rbf':
            param_combinations.extend(product(
                config['train_test_splits'],
                [kernel_value],
                config['C'],
                config['epsilon'],
                config['tol'],
                config['gamma'],
                [None],  # Degree not used
                [None]   # Coef0 not used
            ))
        elif kernel_value == 'poly':
            gamma = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0, 10]
            param_combinations.extend(product(
                config['train_test_splits'],
                [kernel_value],
                config['C'],
                config['epsilon'],
                config['tol'],
                gamma,
                config['degree'],
                config['coef0']
            ))
        elif kernel_value == 'sigmoid':
            param_combinations.extend(product(
                config['train_test_splits'],
                [kernel_value],
                config['C'],
                config['epsilon'],
                config['tol'],
                config['gamma'],
                [None],  # Degree not used
                config['coef0']
            ))
    return param_combinations

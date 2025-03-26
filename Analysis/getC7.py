# this code is working, and it is taking static configuration as config_name for getting the best parameters. HowevergetBestParams_config1.py is dynamic

import os
import shutil
import numpy as np
import pandas as pd

# Base names for the datasets
base_names = ['DSI1', 'DSI2', 'LIB1', 'LIB2', 'MAN2']
base_dir = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/MultiOutput_SVR/Results/plainSVMm_2024"
results_dir = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/MultiOutput_SVR"
train_test_splits = [0.05, 0.1,0.15, 0.2]
data_type = "scaled"
repetitions = 3
max_iter = -1
config_name = "C7"

# Sample configurations (1 through 8), with config9 provided as an example
configurations = {
    "C7": {
        'C': [1.0],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
        'epsilon': [0.001, 0.01, 0.1, 1.0],
        'tol': [0.0001, 0.001, 0.01],
        'degree': [2, 3],
        'coef0': [0, 1, 2, 3],
        'max_iter': max_iter,
        'train_test_splits': train_test_splits,
        'repetitions': repetitions,
        'base_names': base_names,
    }
}

def copy_config_files(results_dir, dataset_path, config_name, base_name, filtered_folders):
    """
    Copies filtered folders from the dataset path to the configured results directory.
    
    Args:
        results_dir (str): Base directory for results.
        dataset_path (str): Path containing the dataset folders to copy.
        config_name (str): Name of the configuration (e.g., "C2").
        base_name (str): Base name for organizing the results folder.
        filtered_folders (list): List of folder names to copy.
        
    Returns:
        None
    """
    # Construct the config folder path
    config_folder_path = os.path.join(results_dir, "Results and Analysis", "plainSVRm_2024", config_name, base_name)
    os.makedirs(config_folder_path, exist_ok=True)
    print(f"Created/verified config folder: {config_folder_path}")

    # Iterate through filtered folders and copy them
    for folder_name in filtered_folders:
        folder_path = os.path.join(dataset_path, folder_name)
        target_path = os.path.join(config_folder_path, folder_name)

        # Check if the source folder exists
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        # Remove the target folder if it exists
        if os.path.exists(target_path):
            print(f"Removing existing folder: {target_path}")
            shutil.rmtree(target_path)

        # Copy the folder or file
        if os.path.isdir(folder_path):
            print(f"Copying directory: {folder_path} -> {target_path}")
            shutil.copytree(folder_path, target_path)
        else:
            print(f"Copying file: {folder_path} -> {config_folder_path}")
            shutil.copy2(folder_path, config_folder_path)
    
    print("Operation completed!")

def save_summary_files(best_configurations, base_dir, config_name):
    config_folder_path = os.path.join(results_dir, "Results and Analysis", "plainSVRm_2024", config_name)
    os.makedirs(config_folder_path, exist_ok=True)
    summary1_data = []#(dataset name, mean error, standard deviation)
    summary2_data = []#(dataset name, configuration values, average error, standard deviation)
    
  
    for dataset, result in best_configurations.items():
        if "config_folder" in result:
            ############################# Uncomment this below line, if you want to save the repeating files as well in the results and analysis folder
            #save_repetition_files(config_name, dataset, result['config_folder'], repetitions)

            print(f"Dataset: {dataset}")
            print(f"  Best Configuration from {config_name}: {result['config_folder']}")
            print(f"  Average Error: {result['avg_error']:.4f}")
            print(f"  Standard Deviation: {result['std_dev']:.4f}")
        else:
            print(f"Dataset: {dataset} - {result['message']}")
            
        # For summary1.csv
        summary1_data.append({
            "Dataset": dataset,
            "Mean Error": result['avg_error'],
            "Standard Deviation": result['std_dev']
        })
        
        # For summary2.csv
        config_values = result['config_folder'].split("_")
        config_dict = {
            "Dataset": dataset,
            #"Configuration": result['config_folder'],
            "C": config_values[1].replace("C", ""),
            "Kernel": config_values[2],
            "Gamma": config_values[3] if "g" in config_values[3] else None,# else "N/A", if use N/A tehn it will type g with original value
            "Epsilon": config_values[4].replace("e", ""),
            "Tolerance": config_values[5].replace("tol", ""),
            "Mean Error": result['avg_error'],
            "Standard Deviation": result['std_dev']
        }
        summary2_data.append(config_dict)

    # Create DataFrames for the summaries
    summary1_df = pd.DataFrame(summary1_data)
    summary2_df = pd.DataFrame(summary2_data)
    
    # Save the DataFrames as CSV files in the config folder
    summary1_file_path = os.path.join(config_folder_path, "mean_error_SD.csv")
    summary2_file_path = os.path.join(config_folder_path, "best_combinations.csv")
    
    summary1_df.to_csv(summary1_file_path, index=False)
    summary2_df.to_csv(summary2_file_path, index=False)

    print(f"Summary files saved in {config_folder_path}.")

def process_error_files(folder_path, repetitions):
    errors = []
    
    for i in range(1, repetitions + 1):
        file_path = os.path.join(folder_path, f"error_rep{i}.csv")
        
        print(f"Looking for file: {file_path}")  # Debug line
                
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist.")
            continue

        try:
            df = pd.read_csv(file_path, header=None)
            errors_rep = pd.to_numeric(df[0], errors='coerce')
            errors_rep = errors_rep.dropna().to_numpy()
            if errors_rep.size > 0:
                errors.append(np.mean(errors_rep))
            else:
                print(f"Warning: Non-numeric values found in {file_path}, skipping this file.")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if errors:
        avg_error = np.round(np.mean(errors),3)
        std_dev = np.round(np.std(errors),3)
        return avg_error, std_dev
    else:
        return None, None

def find_best_configurations(base_path, base_names, repetitions, config_name=None):
    best_results = {}
    
    # Ensure config_name is provided
    if config_name is None or config_name not in configurations:
        print("No valid configuration provided.")
        return best_results

    config_params = configurations[config_name]

    for base_name in base_names:
        dataset_path = os.path.join(base_path, base_name)
        
        print(f"Checking dataset path: {dataset_path}")  # Debug line
        
        if not os.path.exists(dataset_path):
            print(f"Dataset folder {dataset_path} does not exist.")
            continue

        best_error = float('inf')
        best_config = None

        available_folders = os.listdir(dataset_path)
        
        ######################################################################################### trying to make kernel specifc choic of the parameters

        # Define kernel-specific parameters
        kernel_params = {
            "linear": ["C", "epsilon", "tol"],
            "rbf": ["C", "gamma", "epsilon", "tol"],
            "sigmoid": ["C", "gamma", "epsilon", "tol", "coef0"],
            "poly": ["C", "gamma", "epsilon", "degree", "tol",  "coef0"]
        }

        # Initialize the expected parameters list
        expected_parameters = []

        # Dynamically build expected folder patterns
        # filtered_folders = []
        # for folder in available_folders:
        #     if folder.endswith(f"_{data_type}"):
        
        for split in config_params['train_test_splits']:
            for c in config_params['C']:
                for kernel in config_params['kernel']:
                    # Start with common parameters
                    folder_name_pattern = f"split{split}_C{c}_{kernel}_"

                    # Add kernel-specific parameters dynamically
                    if kernel == "sigmoid":  # Handle sigmoid kernel
                        for gamma in config_params['gamma']:
                            for beta in config_params.get("coef0", [0]):
                                pattern = (f"{folder_name_pattern}"
                                        f"g{gamma}_"#
                                        f"e{config_params['epsilon'][0]}_"
                                        f"tol{config_params['tol'][0]}_"
                                        f"beta{beta}_")
                                expected_parameters.append(pattern)
                                # print("Generated Patterns:", expected_parameters)
                                # print("Available Folders:", available_folders)

                                
                    elif kernel == "poly":  # Handle polynomial kernelprint(f"Processing poly kernel with gamma={config_params['gamma']}, coef0={config_params['coef0']}, and degree={config_params['degree']}")
                        for gamma in config_params['gamma']:
                            for beta in config_params.get("coef0", [0]):
                                for degree in config_params.get("degree", [3]):
                                    pattern = (f"{folder_name_pattern}"
                                            f"g{gamma}_"
                                            f"e{config_params['epsilon'][0]}_"
                                            f"d{degree}_"
                                            f"tol{config_params['tol'][0]}_"
                                            f"beta{beta}_")
                                    expected_parameters.append(pattern)

                    else:  # Handle other kernels (linear, rbf, etc.)
                        if "gamma" in kernel_params[kernel]:
                            for gamma in config_params['gamma']:
                                pattern = (f"{folder_name_pattern}"
                                        f"g{gamma}_"
                                        f"e{config_params['epsilon'][0]}_"
                                        f"tol{config_params['tol'][0]}")
                                expected_parameters.append(pattern)
                        else:
                            # Append pattern with common parameters only
                            pattern = (f"{folder_name_pattern}"
                                    f"e{config_params['epsilon'][0]}_"
                                    f"tol{config_params['tol'][0]}")
                            expected_parameters.append(pattern)
        
       
                # Match the folder against expected parameters
                # if any(param in folder for param in expected_parameters):
                #     filtered_folders.append(folder)
        filtered_folders = []
        with os.scandir(dataset_path) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name.endswith(f"_{data_type}"):
                    if any(param in entry.name for param in expected_parameters):
                        filtered_folders.append(entry.name)
        
        #copy_config_files(results_dir, dataset_path, config_name, base_name, filtered_folders)
        ######################################################################################## trying to make kernel specific parameter choice in the above code
        for config_folder in filtered_folders:
           full_path = os.path.join(dataset_path, config_folder)
           if not os.path.isdir(full_path):
               continue

           avg_error, std_dev = process_error_files(full_path, repetitions)
           if avg_error is not None and avg_error < best_error:
               best_error = avg_error
               best_config = {
                   "config_folder": config_folder,
                   "avg_error": avg_error,
                   "std_dev": std_dev
               }

        if best_config:
           best_results[base_name] = best_config
        else:
           best_results[base_name] = {"message": "No valid data found."}

    return best_results


# Call the function with a specific configuration name (e.g., "config9")
best_configurations = find_best_configurations(base_dir, base_names, repetitions, config_name=config_name)
save_summary_files(best_configurations, base_dir, config_name)

# # Print results
# for dataset, result in best_configurations.items():
#     if "config_folder" in result:
#        print(f"Dataset: {dataset}")
#        print(f"  Best Configuration from {config_name}: {result['config_folder']}")
#        print(f"  Average Error: {result['avg_error']:.4f}")
#        print(f"  Standard Deviation: {result['std_dev']:.4f}")
#     else:
#        print(f"Dataset: {dataset} - {result['message']}")


#updated function for taking average for both output files like error_rep*.csv an dprediciton_rep*.csv process_error_files_and_predictions
# 
# import os
# import pandas as pd
# import numpy as np

# def process_error_files_and_predictions(folder_path, repetitions):
#     errors = []
#     mean_predictions = {'Longitude': [], 'Latitude': [], 'Altitude': []}
    
#     for i in range(1, repetitions + 1):
#         # Process error files
#         error_file_path = os.path.join(folder_path, f"error_rep{i}.csv")
#         print(f"Looking for error file: {error_file_path}")  # Debug line
        
#         if not os.path.exists(error_file_path):
#             print(f"Warning: {error_file_path} does not exist.")
#             continue

#         try:
#             df = pd.read_csv(error_file_path, header=None)
#             errors_rep = pd.to_numeric(df[0], errors='coerce')
#             errors_rep = errors_rep.dropna().to_numpy()
#             if errors_rep.size > 0:
#                 errors.append(np.mean(errors_rep))
#             else:
#                 print(f"Warning: Non-numeric values found in {error_file_path}, skipping this file.")
#         except Exception as e:
#             print(f"Error reading {error_file_path}: {e}")
        
#         # Process prediction files
#         prediction_file_path = os.path.join(folder_path, f"predictions_rep{i}.csv")
#         print(f"Looking for prediction file: {prediction_file_path}")  # Debug line
        
#         if not os.path.exists(prediction_file_path):
#             print(f"Warning: {prediction_file_path} does not exist.")
#             continue

#         try:
#             pred_df = pd.read_csv(prediction_file_path)
#             if {'Longitude', 'Latitude', 'Altitude'}.issubset(pred_df.columns):
#                 mean_predictions['Longitude'].append(np.mean(pred_df['Longitude']))
#                 mean_predictions['Latitude'].append(np.mean(pred_df['Latitude']))
#                 mean_predictions['Altitude'].append(np.mean(pred_df['Altitude']))
#             else:
#                 print(f"Warning: Missing columns in {prediction_file_path}, skipping this file.")
#         except Exception as e:
#             print(f"Error reading {prediction_file_path}: {e}")

#     # Calculate average error and standard deviation
#     if errors:
#         avg_error = np.mean(errors)
#         std_dev = np.std(errors)
#     else:
#         avg_error = None
#         std_dev = None
    
#     # Calculate average predictions
#     avg_predictions = {}
#     if mean_predictions['Longitude']:
#         avg_predictions['Longitude'] = np.mean(mean_predictions['Longitude'])
#         avg_predictions['Latitude'] = np.mean(mean_predictions['Latitude'])
#         avg_predictions['Altitude'] = np.mean(mean_predictions['Altitude'])
#     else:
#         avg_predictions = None

#     return avg_error, std_dev, avg_predictions


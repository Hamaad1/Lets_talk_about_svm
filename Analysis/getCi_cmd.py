# Go to ReadMe.txt to understand the code : ../Lets_talk_about_svm/readme.txt
#Command: python getCi_cmd.py --config_name C1 --datasets DSI2 MAN2 --subfolders plainSVRs_2024 plainSVRm_2024

import os
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

    
def copy_config_files(results_dir, dataset_path, config_name, base_name, filtered_folders, config_folder_path):
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

    os.makedirs(config_folder_path, exist_ok=True)
    # print(f"Created/verified config folder: {config_folder_path}")

    for folder_name in filtered_folders:
        folder_path = os.path.join(dataset_path, folder_name)
        target_path = os.path.join(config_folder_path, folder_name)

        # Check if the source folder exists
        if not os.path.exists(folder_path):
            #print(f"Folder not found: {folder_path}")
            continue

        # Remove the target folder if it exists
        if os.path.exists(target_path):
            # print(f"Removing existing folder: {target_path}")
            # shutil.rmtree(target_path)
            print(f"Skipping already processed folder: {target_path}")
            continue

        # Copy the folder or file
        if os.path.isdir(folder_path):
            #print(f"Copying directory: {folder_path} -> {target_path}")
            shutil.copytree(folder_path, target_path)
        else:
            #print(f"Copying file: {folder_path} -> {config_folder_path}")
            shutil.copy2(folder_path, config_folder_path)
    
    #print("Operation completed!")


def save_summary_files(best_configurations, results_dir, config_name, subfolder):
    
    # Initialize data for the summaries
    summary1_data = []  # (dataset name, mean error, standard deviation)
    summary2_data = []  # (dataset name, configuration values, average error, standard deviation)
    
    for dataset, result in best_configurations.items():
        if "config_folder" in result:
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
            "C": config_values[1].replace("C", ""),
            "Kernel": config_values[2],
            "Gamma": config_values[3] if "g" in config_values[3] else "N/A",
            "Epsilon": config_values[4].replace("e", ""),
            "Tolerance": config_values[5].replace("tol", ""),
            "Mean Error": result['avg_error'],
            "Standard Deviation": result['std_dev']
        }
        summary2_data.append(config_dict)

    # Create DataFrames for the summaries
    summary1_df = pd.DataFrame(summary1_data)
    summary2_df = pd.DataFrame(summary2_data)
    
    # Define subfolder-based path
    subfolder_results_dir = os.path.join(results_dir, subfolder, config_name)
    os.makedirs(subfolder_results_dir, exist_ok=True)
    
    # Save the DataFrames as CSV files in the respective subfolder
    summary1_file_path = os.path.join(subfolder_results_dir, "mean_error_SD.csv")
    summary2_file_path = os.path.join(subfolder_results_dir, "best_combinations.csv")
    # Check if files already exist; if so, append new data
    if os.path.exists(summary1_file_path):
        summary1_df.to_csv(summary1_file_path, mode='a', header=False, index=False)
    else:
        summary1_df.to_csv(summary1_file_path, index=False)

    if os.path.exists(summary2_file_path):
        summary2_df.to_csv(summary2_file_path, mode='a', header=False, index=False)
    else:
        summary2_df.to_csv(summary2_file_path, index=False)

    
def process_error_files(folder_path, repetitions):
    errors = []
        
    for i in range(1, repetitions + 1):
        file_path = os.path.join(folder_path, f"error_rep{i}.csv")
    #     print(f"Looking for file: {file_path}")  # Debug line
                
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist.")
            continue
        # else:
        #     print(f"File found: {file_path}")

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
        #print(f"Checking dataset path: {dataset_path}")  # Debug line
        
        if not os.path.exists(dataset_path):
            print(f"Dataset folder {dataset_path} does not exist.")
            continue

        best_error = float('inf')
        best_config = None

        kernel_params = {
            "linear": ["C", "epsilon", "tol"],
            "rbf": ["C", "gamma", "epsilon", "tol"],
            "sigmoid": ["C", "gamma", "epsilon", "tol", "coef0"],
            "poly": ["C", "gamma", "epsilon", "degree", "tol",  "coef0"]
        }

        # Initialize the expected parameters list
        expected_parameters = []
        for split in config_params['train_test_splits']:
            for c in config_params['C']:
                for kernel in config_params['kernel']:
                    folder_name_pattern = f"split{split}_C{c}_{kernel}_"
                    if kernel == "sigmoid":
                        for gamma in config_params['gamma']:
                            for beta in config_params.get("coef0", [0]):
                                pattern = (f"{folder_name_pattern}"
                                        f"g{gamma}_"
                                        f"e{config_params['epsilon'][0]}_"
                                        f"tol{config_params['tol'][0]}_"
                                        f"beta{beta}_")
                                expected_parameters.append(pattern)
                    elif kernel == "poly": 
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
                    else: 
                        if "gamma" in kernel_params[kernel]:
                            for gamma in config_params['gamma']:
                                pattern = (f"{folder_name_pattern}"
                                        f"g{gamma}_"
                                        f"e{config_params['epsilon'][0]}_"
                                        f"tol{config_params['tol'][0]}")
                                expected_parameters.append(pattern)
                        else:
                            pattern = (f"{folder_name_pattern}"
                                    f"e{config_params['epsilon'][0]}_"
                                    f"tol{config_params['tol'][0]}")
                            expected_parameters.append(pattern)
     
        data_type = "scaled"
        filtered_folders = []
        with os.scandir(dataset_path) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name.endswith(f"_{data_type}"):
                    if any(param in entry.name for param in expected_parameters):
                        filtered_folders.append(entry.name)
                        
        #copy_config_files(results_dir, dataset_path, config_name, base_name, filtered_folders, config_folder_path)
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

    
if __name__ == "__main__":
    base_dir = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Results_simulations"
    results_dir = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Results and Analysis"
    train_test_splits = [0.05, 0.1, 0.2]
    max_iter = -1
    repetitions = 2

    os.makedirs(results_dir, exist_ok=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process configurations and datasets for model evaluation.')
    parser.add_argument('--config_name', type=str, required=True, help='Name of the configuration to use.')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of dataset names.')
    parser.add_argument('--subfolders', type=str, nargs='+', required=True, help='List of subfolder names to analyze.')
    parser.add_argument('--repetitions', type=int, default=10, help='Number of repetitions for error computation.')
    
    args = parser.parse_args()

    # Extract arguments
    config_name = args.config_name
    base_names = args.datasets
    subfolders = args.subfolders
    repetitions = repetitions

    # Validate subfolders
    subfolder_paths = [os.path.join(base_dir, subfolder) for subfolder in subfolders]
    for subfolder_path in subfolder_paths:
        if not os.path.exists(subfolder_path):
            raise ValueError(f"The subfolder path '{subfolder_path}' does not exist.")

   
    configurations = {
        "C1": {
            'train_test_splits': [0.2],
            'C': [1.0],
            'kernel': ['rbf'],
            'gamma': ['scale'],
            'epsilon': [0.001],
            'tol': [0.001],
            'repetitions': repetitions
        },
        "C2": {
            'train_test_splits': train_test_splits,
            'C': [1.0],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale'],
            'epsilon': [0.001],
            'degree': [3],
            'coef0': [0.0],
            'tol': [0.001],
            'repetitions': repetitions
        },
        "C3" : {
            'C': [1.0],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale'],
            'coef0': [0, 1, 2, 3], 
            'degree': [2, 3, 4, 5], 
            'tol': [0.001],
            'epsilon': [0.001],
            'train_test_splits': train_test_splits,
            'repetitions': repetitions,
        },


        "C4" : {
            'C': [1.0],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100], 
            'degree': [3],
            'coef0': [0],
            'epsilon': [0.001],
            'tol': [0.001],
            'train_test_splits': train_test_splits,
            'repetitions': repetitions,
        },

        "C5" : {
            'C': [1.0],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto',0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
            'degree': [2, 3, 4], 
            'coef0': [0, 1, 2,3],  
            'tol': [0.001],
            'epsilon': [0.001],
            'train_test_splits': train_test_splits,
            'repetitions':repetitions,
        },
        
        "C6" : {
            'C':[1.0],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
            'degree': [2, 3],
            'coef0' : [0,1,2,3],
            'epsilon': [0.001],
            'tol': [0.0001, 0.001, 0.01],   
            'train_test_splits': train_test_splits,
            'repetitions':repetitions,
            'max_iter': max_iter,
            'base_names': base_names,
        },

        
        "C7" : {
            'C': [1.0],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
            'epsilon': [0.001, 0.01, 0.1, 1.0],
            'tol': [0.0001, 0.001, 0.01], 
            'degree': [2, 3],
            'coef0' : [0,1,2,3],
            'max_iter': max_iter,
            'train_test_splits': train_test_splits,
            'repetitions':repetitions,
            'base_names': base_names,
        },
        
        "C8" : {
            'C': np.linspace(0.001, 1.0, num=5).tolist(),#+ list(range(10, 101,10)),# + list(range(100, 1000, 10)),# Varying C values   # + list(range(1, 100,5)),# + list(range(100, 1000, 10))#np.logspace(-3, 3, 649)
            'kernel': ['rbf'],#'linear','rbf', 'sigmoid', 'poly'], 
            'gamma': ['scale', 'auto', 0.001],#, 0.01, 0.1, 1.0, 10, 100],# 0.0001,
            'epsilon': [0.001, 0.01],#, 0.1, 1.0],
            'tol': [0.0001, 0.001],#, 0.01],
            'degree': [2, 3],
            'coef0' : [0,1],#,2,3],
            'max_iter': max_iter,
            'train_test_splits': train_test_splits,
            'repetitions':repetitions,
            'base_names': base_names,
        }

    }

    # Ensure the configuration exists
    if config_name not in configurations:
        raise ValueError(f"Configuration '{config_name}' not found. Available configurations: {list(configurations.keys())}")

    selected_config = configurations[config_name]

    #for base_name in base_names:
    for subfolder_path, subfolder in zip(subfolder_paths, subfolders):
        for base_name in base_names:
            print(f"Processing Dataset {base_name} in {subfolder} from subfolders: {subfolders} ")
            config_folder_path = os.path.join(results_dir, subfolder, config_name, base_name)
            if os.path.exists(config_folder_path):
                print(f"Skipping already processed folder: {config_folder_path}")
                os.makedirs(config_folder_path, exist_ok=True)

            # Run analysis for the subfolder
            best_configurations = find_best_configurations(
                base_path=subfolder_path,
                base_names=[base_name],#selected_config['base_names'],
                repetitions=selected_config['repetitions'],
                config_name=config_name,
            )
            save_summary_files(best_configurations, results_dir, config_name, subfolder)

#    Command :  python getCi_cmd.py --config_name C1 C2 or any --datasets DSI2 MAN2 --subfolders plainSVRs_2024 plainSVRm_2024
    
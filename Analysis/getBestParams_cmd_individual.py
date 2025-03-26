
# go to Readme.txt at /University_valencia_IP/Lets_talk_about_svm/Analysis/ReadMe.txt

import os
import argparse
import numpy as np
import pandas as pd

# Base names for the datasets
base_names = ['DSI1', 'DSI2', 'LIB1', 'LIB2']
train_test_splits = [0.05, 0.1, 0.2]
data_type = "scaled"
repetitions = 2
max_iter = -1


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
        avg_error = np.mean(errors)
        std_dev = np.std(errors)
        return avg_error, std_dev
    else:
        return None, None


def given_configuration_output(base_path, base_names, repetitions, **kwargs):
    """
    Finds the best configuration for datasets based on specified parameters.

    Args:
        base_path (str): Path to the base directory containing dataset folders.
        base_names (list): List of dataset names.
        repetitions (int): Number of repetitions to consider for error processing.
        **kwargs: Additional parameters such as 'kernel', 'C', 'gamma', etc.

    Returns:
        dict: Best configurations with average error and standard deviation for each dataset.
    """
    config_results  = {}

    for base_name in base_names:
        dataset_path = os.path.join(base_path, base_name)
        if not os.path.exists(dataset_path):
            print(f"Dataset folder {dataset_path} does not exist.")
            continue

        #best_error = float('inf')
        dataset_config_results   = {} 

        available_folders = os.listdir(dataset_path)
        filtered_folders = []

        # Generate expected patterns for folder names
        for folder in available_folders:
            if folder.endswith(f"_{kwargs.get('data_type', '')}"):
                expected_patterns = []

                for split in kwargs.get('test_size', [0.2]):
                    for c in kwargs.get('C', [1.0]):
                        for kernel in kwargs.get('kernel', ['rbf']):
                            # Start with common parameters
                            folder_name_pattern = f"split{split}_C{c}_{kernel}_"

                            if kernel == "poly":
                                for gamma in kwargs.get('gamma', ['scale']):
                                    for coef0 in kwargs.get('coef0', [0]):
                                        coef0 = int(coef0)  
                                        for degree in kwargs.get('degree', [3]):
                                            degree = int(degree)  
                                            pattern = (f"{folder_name_pattern}"
                                                f"g{gamma}_"
                                                f"e{kwargs['epsilon'][0]}_"
                                                f"d{degree}_"
                                                f"tol{kwargs['tol'][0]}_"
                                                f"beta{coef0}_")
                                            expected_patterns.append(pattern)
                            elif kernel == "sigmoid":  # Sigmoid kernel
                                for gamma in kwargs.get('gamma', ['scale']):
                                    for coef0 in kwargs.get('coef0', [0]):
                                        coef0 = int(coef0) 
                                        pattern = (f"{folder_name_pattern}"
                                            f"g{gamma}_"
                                            f"e{kwargs['epsilon'][0]}_"
                                            f"tol{kwargs['tol'][0]}_"
                                            f"beta{coef0}_")
                                        expected_patterns.append(pattern)
                            elif kernel == "rbf":  
                                for gamma in kwargs.get('gamma', ['scale']):
                                        pattern = (f"{folder_name_pattern}"
                                                f"g{gamma}_"
                                                f"e{kwargs['epsilon'][0]}_tol{kwargs['tol'][0]}")
                                        expected_patterns.append(pattern)
                            
                            elif kernel == "linear":
                                    pattern = (f"{folder_name_pattern}"
                                            f"e{kwargs['epsilon'][0]}_tol{kwargs['tol'][0]}")
                                    expected_patterns.append(pattern)

                # Match folder names
                if any(expected in folder for expected in expected_patterns):
                    filtered_folders.append(folder)

        if not filtered_folders:
            print(f"No matching folders found for dataset: {base_name}")
            continue

        # Find the best configuration from filtered folders
        for config_folder in filtered_folders:
            full_path = os.path.join(dataset_path, config_folder)
            if not os.path.isdir(full_path):
                continue

            avg_error, std_dev = process_error_files(full_path, repetitions)
            # Store the results for this configuration
            if avg_error is not None:
                dataset_config_results[config_folder] = {
                    "avg_error": np.round(avg_error,3),
                    "std_dev": np.round(std_dev,3)
                }

            # After processing all configurations for the dataset, store the results
            if dataset_config_results:
                config_results[base_name] = dataset_config_results
            else:
                config_results[base_name] = {"message": "No valid data found."}        

    return config_results



def runmethod_cmd(data_directory, results_directory, log_file, base_name, C, kernel, gamma, epsilon, tol, coef0, degree, test_size, repetitions, max_iter, subfolders):#, Config: {config}
    base_names = [base_name.upper()] 
    print(f"Running with the following configuration:\n"
          f"Base Name: {base_name}\n"
          f"Split: {test_size}, C: {C}, Kernel: {kernel}, Gamma: {gamma}, Epsilon: {epsilon}\n"
          f"Tol: {tol}, degree: {degree}, beta: {coef0}, Repetitions: {repetitions}\n"#, Config: {config}
          f"Data Directory: {data_directory}, Results Directory: {results_directory}\n"
          f"Log File: {log_file}, Max Iter: {max_iter}\n")

    results = given_configuration_output(
        base_path=data_directory,
        base_names=base_names,
        test_size = [test_size],
        C=[C],
        kernel=[kernel],
        gamma=[gamma],
        epsilon=[epsilon],
        tol=[tol],
        coef0=[coef0],
        degree=[degree],
        repetitions=repetitions,
        subfolders=[subfolders],
        data_type="scaled"  
        
    )
    
    result_data = []

    # Output results after processing all datasets
    for base_name, results in results.items():
        print(f"Results for dataset: {base_name}")
        if isinstance(results, dict):
            if "message" in results:
                print(results['message'])
            else:
                for config, result in results.items():
                    print(f"Configuration: {config}")
                    print(f"Average Error: {result['avg_error']}")
                    print(f"Standard Deviation: {result['std_dev']}")
                    # Append to result list for saving
                    result_data.append({
                        "Base Name": base_name,
                        "Configuration": config,
                        "Average Error": result["avg_error"],
                        "Standard Deviation": result["std_dev"]
                    })
        else:
            print(results['message'])
    # Save to CSV file
    if result_data:
        os.makedirs(results_directory, exist_ok=True)
        output_file = os.path.join(results_directory, f"{base_name}_results.csv")
        df = pd.DataFrame(result_data)
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
                    

if __name__ == "__main__":
    # Default paths for directories and files
    data_directory = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Results_simulations"
    results_directory = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Results_individual"
    log_file = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Results SVRm/executed_configs.log'

    os.makedirs(results_directory, exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Run SVR experiments.')
    parser.add_argument('-ts','--test_size', type=float, help='Train-test split ratio.')
    parser.add_argument('-DS','--base_name', type=str, help='Name of the base dataset.')
    parser.add_argument('-C','--C', type=float, help='Regularization parameter C for SVR.')
    parser.add_argument('-K','--kernel', type=str, help='Kernel type to be used (e.g., rbf, linear, sigmoid, poly).')
    parser.add_argument('-g','--gamma', type=str, help='Gamma parameter for the kernel.')
    parser.add_argument('-e','--epsilon', type=float, help='Epsilon parameter for SVR.')
    parser.add_argument('-d','--degree', type=float, help='Tolerance for stopping criterion.')
    parser.add_argument('-t','--tol', type=float, help='Tolerance for stopping criterion.')
    parser.add_argument('-coe','--coef0', type=float, help='Epsilon parameter for SVR.')
    #parser.add_argument('config', type=int, help='Configuration choice (1 or 2).')
    parser.add_argument('-rep','--repetitions', type=int, help='Number of repetitions for the experiment.')
    parser.add_argument('-fold','--subfolders', type=str, nargs='+', required=True, help='List of subfolder names to analyze.')
    parser.add_argument('--data_directory', type=str, default=data_directory, help='Path to the data directory.')
    parser.add_argument('--results_directory', type=str, default=results_directory, help='Path to the results directory.')
    parser.add_argument('--log_file', type=str, default=log_file, help='Path to the executed configurations log file.')
    parser.add_argument('--max_iter', type=int, default=-1, help='Maximum Iteration for convergence.')

    args = parser.parse_args()
    
   # Validate data directory
    if not os.path.exists(data_directory):
        raise ValueError(f"The data directory '{data_directory}' does not exist.")

    # Create results directory based on subfolders
    selected_subfolder = args.subfolders[0]  # Assuming one folder is selected for configuration
    results_directory = os.path.join(args.results_directory, selected_subfolder)
    os.makedirs(results_directory, exist_ok=True)   
    print(f"Results directory dynamically set to: {results_directory}")
    
    # Loop through subfolders
    for subfolder in args.subfolders:
        # Construct the full path for each subfolder
        current_data_directory = os.path.join(data_directory, subfolder)
        
        # Validate subfolder
        if not os.path.exists(current_data_directory):
            raise ValueError(f"The subfolder path '{current_data_directory}' does not exist.")
        
        print(f"Running experiment for subfolder: {current_data_directory}")
        
        runmethod_cmd(
            data_directory=current_data_directory,
            results_directory=results_directory,
            log_file=log_file,
            test_size=args.test_size,
            base_name=args.base_name,
            C=args.C,
            kernel=args.kernel,
            gamma=args.gamma,
            epsilon=args.epsilon,
            degree=args.degree,      
            tol=args.tol,
            coef0=args.coef0,
            repetitions=args.repetitions,
            subfolders=args.subfolders,
            max_iter=args.max_iter
         )

# first get int o the main folder by cd "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm"   
#command: python "Analysis/getBestParams_cmd_individual.py" -fold plainSVRm_2024 -ts 0.2 -DS DSI2 -C 1.0 -K rbf -g scale -e 0.001 -d 3 -t 0.001 -coe 0.0 -rep 2



import os
import sys
import time
import json
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from preprocessing import minmaxscaler, process_datasets
from helping_functions_svm import fit_and_evaluate_model, prepare_data, save_results, save_progress, load_progress, files_exist, generate_param_combinations
import logging
import argparse


dataset_results = []
summary_list = []
best_configs_dict = {}
base_random_state = 220

# Path to store the last processed configuration
progress_file = "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm"

def runSVRm(processed_data, base_names, results_directory, dataset_params, param_combinations, repetitions):#, config, param_combinations):
    max_iter = -1#config['max_iter']
    #repetitions = config.get('repetitions', 10)

    # Prepare data
    X, y, X_testing, y_testing, rsamples1, osamples1, nmacs1, rsamples, osamples, nmacs, newNonDetectedValue, minValueDetected, defaultNonDetectedValue = prepare_data(dataset_params, processed_data)

    # List to save errors and other data
    all_errors_val = []
    all_errors_testing = []
    timing_info = []
    
    best_degree, best_beta, best_split, best_kernel, best_gamma, best_C, best_epsilon, best_tol = [None] * 8
    best_error, best_val_error = float('inf'), float('inf')
    data_type = "scaled"

    # Load the last processed configuration (if any)
    last_processed = load_progress()
    processed_combinations = set()  # Store processed combinations
    
    # if last_processed:
    #     for item in last_processed:  # Assuming `load_progress` returns a list of processed configurations
    #         processed_combinations.add((item['dataset'], item['rep'], tuple(item['params'])))

    # print(f"Resuming from {len(processed_combinations)} previously processed configurations.")

    # Define the total number of configurations
    total_configurations = len(param_combinations) * repetitions
 
    random_state_matrix = {}
    for dataset_index, dataset in enumerate([base_names]):  
        random_states = [base_random_state + 3310 * dataset_index + 71 * (i ** 3) for i in range(repetitions)]
        random_state_matrix[dataset] = random_states

    for dataset in [base_names]:
        random_states = random_state_matrix[dataset]
        #print(f"Processing {len(random_states)} repetitions for dataset: {dataset}")
        for rep, random_state in enumerate(random_states):
            print(f"Running iteration {rep + 1}/{len(random_states)} for dataset: {dataset} with: {len(param_combinations)} parameter combinations")

            for params in tqdm(param_combinations, desc=f"Processing dataset: {dataset}"):
                if last_processed:
                    for item in last_processed:  # Assuming `load_progress` returns a list of processed configurations
                        processed_combinations.add((item['dataset'], item['rep'], tuple(item['params'])))

                print(f"Resuming from {len(processed_combinations)} previously processed configurations.")

                test_split, kernel_value, C_value, epsilon_value, tol_value, gamma_value, degree_value, coef0_value = params
                
                # Train-test split
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, random_state=random_state)
                X_train_scaled, X_val_scaled, X_testing_scaled = minmaxscaler(X_train, X_val, X_testing)
                
                # Determine folder structure based on kernel type
                if kernel_value == 'linear':
                    folder = f"{results_directory}/{dataset}/split{test_split}_C{C_value}_{kernel_value}_e{epsilon_value}_tol{tol_value}_{data_type}"
                elif kernel_value == 'rbf':
                    folder = f"{results_directory}/{dataset}/split{test_split}_C{C_value}_{kernel_value}_g{gamma_value}_e{epsilon_value}_tol{tol_value}_{data_type}"
                elif kernel_value == 'sigmoid':
                    folder = f"{results_directory}/{dataset}/split{test_split}_C{C_value}_{kernel_value}_g{gamma_value}_e{epsilon_value}_tol{tol_value}_beta{coef0_value}_{data_type}"
                elif kernel_value == 'poly':
                    folder = f"{results_directory}/{dataset}/split{test_split}_C{C_value}_{kernel_value}_g{gamma_value}_e{epsilon_value}_d{degree_value}_tol{tol_value}_beta{coef0_value}_{data_type}"
                
                # Ensure folder exists before checking files
                if not os.path.exists(folder):
                    os.makedirs(folder)
                            
                # Skip if this configuration was already processed
                if (dataset, rep + 1, tuple(params)) in processed_combinations and files_exist(folder, rep):
                    print(f"Skipping previously processed configuration: {params}")
                    continue  # Skip to the next iteration
            
                try:
                    # Initialize model based on kernel
                    if kernel_value == 'linear':
                        model = SVR(C=C_value, kernel=kernel_value, epsilon=epsilon_value, tol=tol_value, max_iter=max_iter)
                        model = MultiOutputRegressor(model, n_jobs=-1)

                        # Fit and evaluate the model
                        fit_time, pred_time, mean_3d_error_val, mean_3d_error_testing, errors_testing, y_pred_testing = fit_and_evaluate_model(
                            model=model,
                            X_train_scaled=X_train_scaled,
                            y_train=y_train,
                            X_val_scaled=X_val_scaled,
                            y_val=y_val,
                            X_testing_scaled=X_testing_scaled,
                            y_testing=y_testing
                        )

                        # Print and save results
                        print(f"Current Repetition: {rep+1}/{repetitions} for Parameters: {params}")
                        print(f"Validation error: {mean_3d_error_val:.5f}, Testing error: {mean_3d_error_testing:.5f}")

                        # Save errors and timing info as you have in your original code
                        errors_df = pd.DataFrame({
                            'Sample_Index': range(len(errors_testing)),
                            'Repetition': rep + 1,
                            'Error': np.round(errors_testing, 5)
                        })
                        
                        timing_info.append({
                            'Repetition': rep + 1,
                            'Test size': test_split,
                            'C': C_value,
                            'Kernel': kernel_value,
                            'Training time': fit_time,
                            'Inference time': pred_time,
                            'Validation error': mean_3d_error_val,
                            'Testing error': mean_3d_error_testing
                        })

                        # # Create the split folder and tol folder in one step
                        # split_folder = os.path.join(results_directory, base_names)
                        # folder = os.path.join(split_folder, f"split{test_split}_C{C_value}_{kernel_value}_e{epsilon_value}_tol{tol_value}_{data_type}")
                        
                        individual_errors_file, predictions_file = save_results(
                            folder=folder, rep=rep, errors_testing=errors_testing, y_pred_testing=y_pred_testing
                        )
                        
                        # Save progress incrementally after each configuration
                        processed_combinations.add((dataset, rep, tuple(params))) # Mark this combination as processed
                        save_progress({'dataset': dataset, 'rep': rep+1, 'params': params})
                    
                        # Update best configuration
                        if mean_3d_error_testing < best_error:
                            best_error = mean_3d_error_testing
                            best_kernel = kernel_value
                            best_C = C_value
                            best_epsilon = epsilon_value
                            best_split = test_split
                            best_tol = tol_value
                            best_val_error = mean_3d_error_val   
            
                    elif kernel_value == 'rbf':
                        model = SVR(C=C_value, kernel=kernel_value, epsilon=epsilon_value, gamma=gamma_value, tol=tol_value, max_iter=max_iter)
                        model = MultiOutputRegressor(model, n_jobs=-1)

                        # Fit and evaluate the model
                        fit_time, pred_time, mean_3d_error_val, mean_3d_error_testing, errors_testing, y_pred_testing = fit_and_evaluate_model(
                            model=model,
                            X_train_scaled=X_train_scaled,
                            y_train=y_train,
                            X_val_scaled=X_val_scaled,
                            y_val=y_val,
                            X_testing_scaled=X_testing_scaled,
                            y_testing=y_testing
                        )

                        # Print and save results
                        print(f"Current Repetition: {rep+1}/{repetitions} for Parameters: {params}")
                        print(f"Validation error: {mean_3d_error_val:.5f}, Testing error: {mean_3d_error_testing:.5f}")

                        # Save errors and timing info as you have in your original code
                        errors_df = pd.DataFrame({
                            'Sample_Index': range(len(errors_testing)),
                            'Repetition': rep + 1,
                            'Error': np.round(errors_testing, 5)
                        })
                        timing_info.append({
                            'Repetition': rep + 1,
                            'Test size': test_split,
                            'C': C_value,
                            'Kernel': kernel_value,
                            'Training time': fit_time,
                            'Inference time': pred_time,
                            'Validation error': mean_3d_error_val,
                            'Testing error': mean_3d_error_testing
                        })

                        # # Create the split folder and tol folder in one step
                        # split_folder = os.path.join(results_directory, base_names)
                        # folder = os.path.join(split_folder, f"split{test_split}_C{C_value}_{kernel_value}_g{gamma_value}_e{epsilon_value}_tol{tol_value}_{data_type}")
                        
                        individual_errors_file, predictions_file = save_results(
                            folder=folder, rep=rep, errors_testing=errors_testing, y_pred_testing=y_pred_testing
                        )
                        
                        # Save progress incrementally after each configuration
                        processed_combinations.add((dataset, rep, tuple(params))) # Mark this combination as processed
                        save_progress({'dataset': dataset, 'rep': rep+1, 'params': params})
                    
                        # Update best configuration
                        if mean_3d_error_testing < best_error:
                            best_error = mean_3d_error_testing
                            best_split = test_split
                            best_C = C_value
                            best_kernel = kernel_value
                            best_gamma = gamma_value
                            best_epsilon = epsilon_value
                            best_tol = tol_value
                            best_val_error = mean_3d_error_val  
                             
                    elif kernel_value == 'sigmoid':
                        model = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, coef0=coef0_value, tol=tol_value, max_iter=max_iter)
                        model = MultiOutputRegressor(model, n_jobs=-1)

                        fit_time, pred_time, mean_3d_error_val, mean_3d_error_testing, errors_testing, y_pred_testing = fit_and_evaluate_model(
                            model=model,
                            X_train_scaled=X_train_scaled,
                            y_train=y_train,
                            X_val_scaled=X_val_scaled,
                            y_val=y_val,
                            X_testing_scaled=X_testing_scaled,
                            y_testing=y_testing
                        )
                        
                        # Print and save results
                        print(f"Current Repetition: {rep+1}/{repetitions} for Parameters: {params}")
                        print(f"Validation error: {mean_3d_error_val:.5f}, Testing error: {mean_3d_error_testing:.5f}")
              
                        # Save individual errors to a CSV file
                        errors_df = pd.DataFrame({
                            'Sample_Index': range(len(errors_testing)),
                            'Repetition': rep + 1,
                            'Error': np.round(errors_testing , 5)
                        })
                        # Save the timing information
                        timing_info.append({
                            'Repetition': rep + 1,
                            'Test size': test_split,
                            'C': C_value,
                            'Kernel': kernel_value,
                            'Training time': fit_time,
                            'Inference time': pred_time,
                            'Validation error': mean_3d_error_val,
                            'Testing error': mean_3d_error_testing
                        })

                        # # Create the split folder and tol folder in one step
                        # split_folder = os.path.join(results_directory, base_names)
                        # folder = os.path.join(split_folder, f"split{test_split}_C{C_value}_{kernel_value}_g{gamma_value}_e{epsilon_value}_tol{tol_value}_beta{coef0_value}_{data_type}")

                        individual_errors_file, predictions_file = save_results(
                            folder=folder, rep=rep, errors_testing=errors_testing, y_pred_testing=y_pred_testing
                        )
                        
                        # Save progress incrementally after each configuration
                        processed_combinations.add((dataset, rep, tuple(params))) # Mark this combination as processed
                        save_progress({'dataset': dataset, 'rep': rep+1, 'params': params})
                    
                        if mean_3d_error_testing < best_error:
                            best_error = mean_3d_error_testing
                            best_kernel = kernel_value
                            best_C = C_value
                            best_gamma = gamma_value
                            best_epsilon = epsilon_value
                            best_split = test_split
                            best_tol = tol_value
                            best_beta = best_beta
                            best_val_error = mean_3d_error_val
  
                    elif kernel_value == 'poly':
                        model = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, degree=degree_value, coef0=coef0_value, tol=tol_value, max_iter=max_iter)        
                        model = MultiOutputRegressor(model, n_jobs=-1)

                        fit_time, pred_time, mean_3d_error_val, mean_3d_error_testing, errors_testing, y_pred_testing = fit_and_evaluate_model(
                            model=model,
                            X_train_scaled=X_train_scaled,
                            y_train=y_train,
                            X_val_scaled=X_val_scaled,
                            y_val=y_val,
                            X_testing_scaled=X_testing_scaled,
                            y_testing=y_testing
                        )
                        
                        # Print and save results
                        print(f"Current Repetition: {rep+1}/{repetitions} for Parameters: {params}")
                        print(f"Validation error: {mean_3d_error_val:.5f}, Testing error: {mean_3d_error_testing:.5f}")
              
                        # print(f"Current Parameters Repetition {rep+1}/{repetitions}, Test size: {test_split}, C:{C_value}, "
                        #     f"Kernel:{kernel_value}, gamma:{gamma_value}, Degree:{degree_value}, epsilon:{epsilon_value}, Beta:{coef0_value}, tol:{tol_value},  "
                        #     f"Validation error: {mean_3d_error_val:.5f}, Testing error: {mean_3d_error_testing:.5f}")
                        
                        # Save individual errors to a CSV file
                        errors_df = pd.DataFrame({
                            'Sample_Index': range(len(errors_testing)),
                            'Repetition': rep + 1,
                            'Error': np.round(errors_testing , 5)
                        })
                        
                        # Save the timing information
                        timing_info.append({
                            'Repetition': rep + 1,
                            'Test size': test_split,
                            'C': C_value,
                            'Kernel': kernel_value,
                            'Training time': fit_time,
                            'Inference time': pred_time,
                            'Validation error': mean_3d_error_val,
                            'Testing error': mean_3d_error_testing
                        })
                        
                        # # Create the split folder and tol folder in one step
                        # split_folder = os.path.join(results_directory, base_names)
                        # folder = os.path.join(split_folder, f"split{test_split}_C{C_value}_{kernel_value}_g{gamma_value}_e{epsilon_value}_d{degree_value}_tol{tol_value}_beta{coef0_value}_{data_type}")
                        
                        individual_errors_file, predictions_file = save_results(
                            folder=folder, rep=rep, errors_testing=errors_testing, y_pred_testing=y_pred_testing
                        )       
                        
                        # Save progress incrementally after each configuration
                        processed_combinations.add((dataset, rep, tuple(params))) # Mark this combination as processed
                        save_progress({'dataset': dataset, 'rep': rep+1, 'params': params})
                        
                        if mean_3d_error_testing < best_error:
                            best_error = mean_3d_error_testing
                            best_kernel = kernel_value
                            best_C = C_value
                            best_tol = tol_value
                            best_gamma = gamma_value
                            best_split = test_split
                            best_epsilon = epsilon_value
                            best_beta = coef0_value
                            best_degree = degree_value
                            best_val_error = mean_3d_error_val
                    else:
                        print(f"Unsupported kernel type: {kernel_value}")
                        continue

                except Exception as e:
                    print(f"Unexpected crash: {e}")
                    #logging.error(f"Error processing: Dataset={dataset}, Rep={rep}, Params={params} | Error: {e}")
                    # traceback.print_exc()
                    print(f"Error processing combination: Dataset={dataset}, Rep={rep}, Params={params} ")
                    logging.error(f"Failed for: Dataset={dataset}, Rep={rep}, Params={params} | Error: {str(e)}")
                    continue  # Skip to the next iteration

    # If all configurations have been processed, notify the user
    if len(processed_combinations) == total_configurations:#all_done:
        print("All configurations for this dataset have already been processed.")
    else:
        print("Some configurations were processed. Please re-run the script to process any remaining ones.")

    # Save timing information to a text file
    with open(os.path.join(results_directory, 'timing_info.txt'), 'w') as f:
        f.write("Repetition\tTest Size\tC\tKernel\tTraining Time (s)\tInference Time (s)\tValidation Error\tTesting Error\n")
        for entry in timing_info:
            f.write(f"{entry['Repetition']}\t{entry['Test size']}\t{entry['C']}\t{entry['Kernel']}\t{entry['Training time']:.4f}\t{entry['Inference time']:.4f}\t{entry['Validation error']:.5f}\t{entry['Testing error']:.5f}\n")

    # Save best configurations as in your original code
    best_configs_dict[base_names] = {
        'C': best_C, 
        'kernel': best_kernel,
        'degree': best_degree,
        'gamma': best_gamma, 
        'epsilon': best_epsilon, 
        'Test split': best_split,
        'Beta': best_beta,
        'tol': best_epsilon,
        'Val_error': best_val_error, 
        'Test_error': best_error
    }
    
    summary_list.append([base_names, best_C, best_kernel, best_degree, best_gamma, best_epsilon, best_split, best_beta, best_tol, best_val_error, best_error])
    summary_file_path = os.path.join(results_directory, 'summary_best_configurations.csv')
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_list, columns=[
        'Dataset', 'Best C', 'Best kernel','Best degree', 'Best gamma', 
        'Best Epsilon', 'Best Split', 'Best Beta', 'Best tol', 
        'Validation Error', 'Testing Error'
    ])

    # Check if the file exists
    if os.path.exists(summary_file_path):
        # If file exists, read existing data and append
        existing_df = pd.read_csv(summary_file_path)
        combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates()

    else:
        # If file does not exist, save the current DataFrame as new
        combined_df = summary_df
    combined_df.to_csv(summary_file_path, index=False)
    
    print(f"\nRunning MultiOutput SVR algorithm on: {base_names}")#{result['dataset']}")
    print(f'    database features pre  : [{rsamples1},{osamples1},{nmacs1}]')
    print(f'    database New features  : [{rsamples},{osamples},{nmacs}]')
    print(f'    Best C                 : {best_C}')
    print(f'    Best kernel            : {best_kernel}')
    print(f'    Best gamma             : {best_gamma}')
    print(f'    Best Epsilon           : {best_epsilon }')
    print(f'    Best Tolerance         : {best_tol}')
    print(f'    Best Beta              : {best_beta}')
    print(f'    Best Degree            : {best_degree}')
    print(f"Avg 3D Positioning Error Validation  : {best_val_error:.5f} m")
    print(f"Avg 3D Positioning Error Testing: {best_error:.5f} m\n")
   

#def plain_SVRm_2024(data_directory, results_directory, args, log_file, max_iter,config, test_size, base_names, C, kernel, gamma, epsilon, tol,  beta, degree, repetitions):
def plain_SVRm_2024(data_directory, results_directory, args, log_file, max_iter, config, test_size, base_names, repetitions):

    for base_name in base_names:
        print(f"Starting dataset processing... {base_name}")
        # Ensure the results directory exists
        os.makedirs(results_directory, exist_ok=True)
        processed_datasets, dataset_params = process_datasets(base_names, data_directory, results_directory)#args.base_names
        param_combinations = generate_param_combinations(config)

        # Run the specified function
        if base_name in processed_datasets.keys():
            data = processed_datasets[base_name]
            base_results_directory = os.path.join(results_directory, 'plainSVRm_2024_test')
            runSVRm(data, base_name, base_results_directory, dataset_params[base_name], param_combinations, repetitions)
        else:
            print(f"Warning: {base_name} not found in processed datasets.")
             
    print("Processing completed for all datasets.")
    
if __name__ =="__main__":
    
    data_directory = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/dataset'
    results_directory = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Results_simulations'
    log_file = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/executed_configs.log'
    max_iter = -1
    
    parser = argparse.ArgumentParser(description="Run exhaustive grid search for SVM model.")
    parser.add_argument('-rep','--repetitions', type=int, default=5, help="Number of repetitions for the experiment.")
    parser.add_argument('-DS','--base_names', nargs='+', help="List of base names of datasets.", required=True)
    parser.add_argument('-C','--C', type=float, nargs='+', default=[1.0], help="SVM regularization parameter.")
    parser.add_argument('-k','--kernel', type=str, nargs='+',default=['rbf'], choices=['linear', 'rbf', 'sigmoid', 'poly'], required=True, help="SVM kernel type.")
    parser.add_argument('-g','--gamma', type=str, nargs='+', default=['scale'], help="Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.")
    parser.add_argument('-e','--epsilon', type=float, nargs='+', default=[0.001], help="Epsilon parameter for SVR.")
    parser.add_argument('-t','--tol', type=float, nargs='+', default=[0.0001], help="Tolerance for stopping criterion.")
    parser.add_argument('-b','--beta', type=float, nargs='+', default=[0.0], help="Value for the beta.")
    parser.add_argument('-d','--degree', type=int, nargs='+', default=[3], help="Degree value for polynomial kernel.")
    parser.add_argument('-ts','--test_size', type=float, nargs='+', default=[0.2], help="Proportion of dataset to be used for testing.")
    
    
    args = parser.parse_args()
    
    config = {
        "C": args.C,
        "kernel": args.kernel,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "tol": args.tol,
        "coef0" : args.beta,
        "degree": args.degree,
        "train_test_splits": args.test_size
    }

    plain_SVRm_2024(
        data_directory=data_directory,
        results_directory=results_directory,
        args=args,  # Pass the entire args object
        log_file=log_file,
        max_iter=max_iter,  # This value is already set, so you can remove it if not needed
        test_size=args.test_size,
        config=config,
        base_names=args.base_names,
        #C=args.C, kernel=args.kernel, gamma=args.gamma, epsilon=args.epsilon, tol=args.tol, coef0= args.beta,degree = args.degree,         
        repetitions=args.repetitions
    )
    
    #command
    #python plain_SVRm_2024.py --test_size 0.1 0.2 --base_names DSI1 MAN2 --C 0.01 1.0 --kernel rbf linear --gamma scale auto --epsilon 0.001 0.01 --tol 0.001 0.01 --beta 0.0 --degree 3 --repetitions 2
    

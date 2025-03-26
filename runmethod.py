# Import necessary libraries
import os
import sys
import json
import traceback
import numpy as np
from itertools import product
from helping_functions_svm import fit_and_evaluate_model, prepare_data, save_results, save_progress, load_progress, files_exist, generate_param_combinations

# Add required path for custom modules
sys.path.append('/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm')

from preprocessing import process_datasets
from plain_SVRm_2024 import runSVRm
from plain_SVRs_2024 import runSVRs

# Define parameters
base_names = ['DSI2', 'MAN2']#'DSI1', 'DSI2','LIB1','LIB2']  # Using only one dataset, modify as needed
train_test_splits = [0.05]#,0.1,0.15,0.2]
repetitions = 3

config = {
    'C': np.linspace(0.001, 1.0, num=5).tolist(),#+ list(range(10, 100,10)) + list(range(150, 1050, 50)),# Varying C values   # + list(range(1, 100,5)),# + list(range(100, 1000, 10))#np.logspace(-3, 3, 649)
    'kernel': ['linear'],#,'rbf', 'sigmoid', 'poly'], 
    'gamma': ['scale', 'auto', 0.001 ],# , 0.01, 0.1, 1.0, 10, 100],
    'epsilon': [0.001, 0.01],#  , 0.1, 1.0],
    'tol': [0.0001, 0.001],#  , 0.01],
    'degree': [2, 3],
    'coef0' : [0,1,2,3],
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'base_names': base_names,
}


def main():
    # Define directories
    data_directory = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/dataset'
    results_directory = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Results_simulations/plainSVRm_2024'

    # Ensure the results directory exists
    os.makedirs(results_directory, exist_ok=True)

    # Process datasets
    print("Starting dataset processing...")
    processed_datasets, dataset_params = process_datasets(base_names, data_directory, results_directory)
    param_combinations = generate_param_combinations(config)
   
    for base_name in processed_datasets.keys():
        data = processed_datasets[base_name]
 
        runSVRm(data, base_name, results_directory, dataset_params[base_name], param_combinations, repetitions)#,config, param_combinations, )

        #runSVRs(data, base_name, results_directory, dataset_params[base_name], param_combinations, repetitions)#config,

    #print("Processing completed for all datasets.")

if __name__ == "__main__":
    main()

import os
import numpy as np
import sys
import json
from itertools import product
import argparse

# Import custom modules
sys.path.append('/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/runmethod_cmd_outputs')
from preprocessing import process_datasets
from plain_SVRm_2024 import runSVRm
from plain_SVRs_2024 import runSVRs

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
            param_combinations.extend(product(
                config['train_test_splits'],
                [kernel_value],
                config['C'],
                config['epsilon'],
                config['tol'],
                config['gamma'],
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

#using belof defination becasue i am passing configuration , not individual parameters
#def runmethod_cmd(data_directory, results_directory, args, log_file, max_iter, test_size, base_names, C, kernel, gamma, epsilon, tol, beta, degree, repetitions):
def runmethod_cmd(data_directory, results_directory, args, log_file, max_iter, config, test_size, base_names, repetitions):

    for base_name in base_names:
        print(f"Processing dataset: {base_name}")
        # Ensure the results directory exists
        os.makedirs(results_directory, exist_ok=True)

        # Process datasets
        print("Starting dataset processing...")
        processed_datasets, dataset_params = process_datasets(args.base_names, data_directory, results_directory)
        param_combinations = generate_param_combinations(config)

        # Run the specified function
        for base_name in processed_datasets.keys():
            data = processed_datasets[base_name]
            if args.function == 'runSVRm':
                results_directory = os.path.join(results_directory, 'plainSVRm_2024')
                runSVRm(data, base_name, results_directory, dataset_params[base_name], param_combinations, repetitions)
            elif args.function == 'runSVRs':
                results_directory = os.path.join(results_directory, 'plainSVRs_2024')
                runSVRs(data, base_name, results_directory, dataset_params[base_name], param_combinations, repetitions)

    print("Processing completed for all datasets.")


if __name__ == "__main__":
    data_directory = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/dataset'
    results_directory = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Results_simulations'
    log_file = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/executed_configs.log'
    max_iter = -1
    
    # config ={
    #     "C": np.linspace(0.001, 1.0, num=5).tolist() + list(range(10, 100,10)) + list(range(150, 1050, 50)),
    #     "kernel": ["linear", "rbf" , "sigmoid", "poly"],
    #     "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
    #     "epsilon": [0.001, 0.01, 0.1, 1.0],
    #     "tol": [0.0001, 0.001, 0.01, 0.1],
    #     "degree": [2, 3],
    #     "coef0": [0, 1, 2,3,4],
    #     "max_iter": -1,
    #     "train_test_splits": [0.05, 0.1,0.15, 0.2,0.25,0.3,0.35,0.4,0.45,0.5],
    #     "repetitions": 10,
    #     "base_names": ["DSI1", "DSI2", "LIB1", "LIB2", "MAN1", "MAN2", "SAH1", "SIM001", "TIE1", "TUT1", "TUT2", "TUT3", "TUT4", "TUT5", "TUT6", "TUT7", "UJI1", "UTS1", "GPR00", "GPR01", "GPR02", "GPR03", "GPR04", "GPR05", "GPR06", "GPR07", "GPR08", "GPR09", "GPR10", "GPR11", "GPR12", "GPR13", "SOD01", "SOD02", "SOD03", "SOD04", "SOD05", "SOD06", "SOD07", "SOD08", "SOD09", "UEXB1", "UEXB2", "UEXB3", "UJIB1", "UJIB2"]
    #     }

    # #parser.add_argument('--config', type=str, required=True, help="Path to the config file.")
    # #parser.add_argument('--data_directory', type=str, required=True, help="Directory containing the datasets.")
    # #parser.add_argument('--results_directory', type=str, required=True, help="Directory to save the results.")
     # #parser.add_argument('--log_file', type=str, required=True, help="Path to the log file.")
    # #parser.add_argument('--max_iter', type=int, default=-1, help="Maximum number of iterations.")
   
    parser = argparse.ArgumentParser(description="Run exhaustive grid search for SVM model.")
    parser.add_argument('--repetitions', type=int, default=5, help="Number of repetitions for the experiment.")
    parser.add_argument('--base_names', nargs='+', help="List of base names of datasets.", required=True)
    parser.add_argument('--C', type=float, nargs='+', default=[1.0], help="SVM regularization parameter.")
    parser.add_argument('--kernel', type=str, nargs='+', choices=['linear', 'rbf', 'sigmoid', 'poly'], required=True, help="SVM kernel type.")
    parser.add_argument('--gamma', type=str, nargs='+', default=['scale'], help="Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.")
    parser.add_argument('--epsilon', type=float, nargs='+', default=[0.001], help="Epsilon parameter for SVR.")
    parser.add_argument('--tol', type=float, nargs='+', default=[0.001], help="Tolerance for stopping criterion.")
    parser.add_argument('--beta', type=float, nargs='+', default=[0.0], help="Value for the beta.")
    parser.add_argument('--degree', type=int, nargs='+', default=[3], help="Degree value for polynomial kernel.")
    parser.add_argument('--test_size', type=float, nargs='+', default=[0.2], help="Proportion of dataset to be used for testing.")
    
    
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

    runmethod_cmd(
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
    
#becasue of nargs='+', i can pass more arguments as in below line
#python plain_SVRm_2024.py --test_size 0.1 0.2 --base_names DSI1 MAN2 --C 0.01 1.0 --kernel rbf linear --gamma scale auto --epsilon 0.001 0.01 --tol 0.001 0.01 --beta 0.0 --degree 3 --repetitions 2






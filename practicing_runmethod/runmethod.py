import os
import sys
import tensorflow as tf
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'practicing_runmethod'))

from preprocessing import process_datasets, minmaxscaler
from configuration1 import runSVR
from Configuration2 import runSVR_C2
from Configuration3 import runSVR_C3
from Configuration4 import runSVR_C4
from Configuration5 import runSVR_C5

base_names = [ 'DSI1', 'DSI2']#, 'LIB1', 'LIB2', 'MAN1', 'MAN2', 'SAH1', 'TIE1', 'TUT1','TUT2', 'TUT3', 'TUT4', 'TUT5', 'TUT6', 'TUT7', 'UJI1', 'UTS1', 'GPR00' , 'GPR01' , 'GPR02', 'GPR03', 'GPR04', 'GPR05', 'GPR06', 'GPR07', 'GPR08', 'GPR09', 'GPR10', 'GPR11', 'GPR12', 'GPR13' ,'SOD01', 'SOD02', 'SOD03', 'SOD04', 'SOD05', 'SOD06', 'SOD07', 'SOD08', 'SOD09', 'UEXB1', 'UEXB2', 'UEXB3', 'UJIB1', 'UJIB2']
max_iter = 5000000
train_test_splits = [0.1, 0.15, 0.2, 0.25]#, 0.3, 0.35, 0.4, 0.45, 0.5
repetitions = 5

config1 = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 0.01,
    'epsilon': 0.001,
    'degree': 3,
    'tol': 0.001,
    'train_test_splits': 0.2,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,

}

config2 = {
    'C': 1.0,
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': 'scale',
    'epsilon': 0.001,
    'degree': [2, 3, 4, 5],
    'tol': 0.001,
    'coef0' : [0,1,2,3],
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,
}

config3 = {
    'C': 1.0,
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.001],
    'epsilon': 0.001,
    'degree': [2, 3, 4, 5],
    'tol': 0.001,
    'coef0' : [0,1,2],
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,
}

config4 = {
    'C': [0.001, 0.01,0.1],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.001]
    'epsilon': 0.001,
    'degree': [2, 3 4, 5],
    'tol': 0.001,
    'coef0' : [0,1,2,3],
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,
}


config5 = {
    'C': [0.001, 0.01,0.1, 10,50,100],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0, 10],
    'epsilon': [0.001, 0.01, 0.1, 1.0] ,
    'degree': [2, 3, 4, 5],
    'tol': [0.0001, 0.001, 0.01],
    'max_iter': max_iter,
    'coef0' : [0,1,2,3],
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'base_names': base_names,
}


# Now update your main function
def main():
    # Define your directories
    data_directory = os.path.join(current_dir, 'practicing_runmethod', 'datasets')
    results_directory = os.path.join(current_dir, 'practicing_runmethod', 'results')
    
    # Prompt user to select a configuration
    print("Select a configuration:")
    print("1: Configuration 1 (runSVR)")
    print("2: Configuration 2 (runSVR2)")
    print("3: Configuration 3 (runSVR3)")
    print("4: Configuration 4 (runSVR4)")
    print("5: Configuration 5 (runSVR5)")
    
    config_choice = input("Enter the configuration number (1-5): ")
    
    # Set the results directory based on selected configuration
    if config_choice == '1':
        results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C1')
        config = config1
    elif config_choice == '2':
        results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C2')
        config = config2
    elif config_choice == '3':
        results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C3')
        config = config3
    elif config_choice == '4':
        results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C4')
        config = config4
    elif config_choice == '5':
        results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C5')
        config = config5
    else:
        print("Invalid choice. Exiting.")
        return

    # Make sure the results directory exists
    os.makedirs(results_directory, exist_ok=True)

    base_names = config['base_names']
    # Process datasets
    print("Starting dataset processing...")
    processed_datasets, dataset_params = process_datasets(base_names, data_directory, results_directory)

    while True:
        print("\nAvailable datasets:")
        for base_name in processed_datasets.keys():
            print(f"- {base_name}")

        print("Type 'all' to process all datasets or enter the name of a specific dataset (or 'q' to quit): ")
        user_choice = input().strip()
        if user_choice.lower() == 'q':
            break

        if user_choice.lower() == 'all':
            # Process all datasets with the specified configuration
            for base_name in processed_datasets.keys():
                data = processed_datasets[base_name]
                if config_choice == '1':
                    print(f"Running Configuration 1 for: {base_name}")
                    runSVR(data, base_name, results_directory, dataset_params[base_name], config)
                elif config_choice == '2':
                    print(f"Running Configuration 2 for: {base_name}")
                    runSVR_C2(data, base_name, results_directory, dataset_params[base_name], config)
                elif config_choice == '3':
                    print(f"Running Configuration 3 for: {base_name}")
                    runSVR_C3(data, base_name, results_directory, dataset_params[base_name], config)
                elif config_choice == '4':
                    print(f"Running Configuration 4 for: {base_name}")
                    runSVR_C4(data, base_name, results_directory, dataset_params[base_name], config)
                elif config_choice == '5':
                    print(f"Running Configuration 5 for: {base_name}")
                    runSVR_C5(data, base_name, results_directory, dataset_params[base_name], config)
                else:
                    print("Invalid configuration choice. Please try again.")
                    break

            print("Data processed successfully for all datasets.")
            break

        else:
            if user_choice not in processed_datasets:
                print("Invalid dataset name. Please try again.")
                continue

            data = processed_datasets[user_choice]
            print(f"Running SVR experiment for: {user_choice}")

            # Process the specific dataset with the selected configuration
            if config_choice == '1':
                runSVR(data, user_choice, results_directory, dataset_params[base_name], config)
            elif config_choice == '2':
                runSVR_C2(data, user_choice, results_directory, dataset_params[base_name], config)
            elif config_choice == '3':
                runSVR_C3(data, user_choice, results_directory, dataset_params[base_name], config)
            elif config_choice == '4':
                runSVR_C4(data, user_choice, results_directory, dataset_params[base_name], config)
            elif config_choice == '5':
                runSVR_C5(data, user_choice, results_directory, dataset_params[base_name], config)

if __name__ == "__main__":
    main()



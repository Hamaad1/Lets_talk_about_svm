#lis tof parameters used by different kernels

# Linear: C, epsilon, tol, max_iter
# Polynomial: C, epsilon, degree, gamma, coef0, tol, max_iter
# RBF: C, epsilon, gamma, tol, max_iter
# Sigmoid: C, epsilon, gamma, coef0, tol, max_iter

import os
import sys
import tensorflow as tf
import numpy as np

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Update the system path to include the 'practicing_runmethod' directory
sys.path.append(os.path.join(current_dir, 'Singleoutput_SVR'))

from preprocessing import process_datasets
from configuration1 import runSVR
from Configuration2 import runSVR_C2
from Configuration3 import runSVR_C3
from Configuration4 import runSVR_C4
from Configuration5 import runSVR_C5
from Configuration6 import runSVR_C6
from Configuration7 import runSVR_C7
from Configuration8 import runSVR_C8
from Configuration9 import runSVR_C9

base_names = [ 'DSI1', 'DSI2']#, 'LIB1', 'LIB2', 'MAN1', 'MAN2', 'SAH1', 'TIE1', 'TUT1','TUT2', 'TUT3', 'TUT4', 'TUT5', 'TUT6', 'TUT7', 'UJI1', 'UTS1', 'GPR00' , 'GPR01' , 'GPR02', 'GPR03', 'GPR04', 'GPR05', 'GPR06', 'GPR07', 'GPR08', 'GPR09', 'GPR10', 'GPR11', 'GPR12', 'GPR13' ,'SOD01', 'SOD02', 'SOD03', 'SOD04', 'SOD05', 'SOD06', 'SOD07', 'SOD08', 'SOD09', 'UEXB1', 'UEXB2', 'UEXB3', 'UJIB1', 'UJIB2']
max_iter = 5000000
train_test_splits = [0.1, 0.15]#, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
repetitions = 5

config1 = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 0.01,
    'epsilon': 0.001,
    'tol': 0.001,
    'train_test_splits': 0.2,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,

}

#this should be the configuration 2, compare kernels with default values
config2 = {
    'C': 1.0,
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': 'scale',
    'epsilon': 0.001,
    'tol': 0.001,
    'degree': 3,
    'coef0' : 0,
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,
}

config3 = {
    'C': 1.0,
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': 'scale',
    'coef0' : [0,1,2,3],
    'degree': [2, 3, 4, 5],
    'tol': 0.001,
    'epsilon': 0.001,
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,
}

#this should be 3, comparing kernels with different gama values
config4 = {
    'C': 1.0,
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.001, 0.01],#, 0.1, 1.0, 10],
    'degree': 3,
    'coef0' : 0,
    'epsilon': 0.001,
    'tol': 0.001,
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,
}
#comparing kerneal with different degree and beta values as well
config5 = {
    'C': 1.0,
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0, 10],
    'degree': [2, 3, 4, 5],
    'coef0' : [0,1,2],
    'tol': 0.001,
    'epsilon': 0.001,
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,
}
#this should be 4th config comparing kernels with different gamma, degree and beta values
config6 = {
    'C':1.0,
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.0000001, 0.000001],#, 0.00001,0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
    'degree': [2, 3, 4, 5],
    'coef0' : [0,1,2,3],
    'epsilon': 0.001,
    'tol': 0.001,
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,
}

#this should be 4th config comparing kernels with different gamma, degree and beta values and tolerance
config7 = {
    'C':1.0,
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.0000001, 0.000001],#, 0.00001,0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
    'degree': [2, 3, 4, 5],
    'coef0' : [0,1,2,3],
    'epsilon': 0.001,
    'tol': [0.0001, 0.001, 0.01],
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'max_iter': max_iter,
    'base_names': base_names,
}

# this should be the 5th config here i am comparing everything with eachother except c
config8 = {
    'C': 1.0,
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.0000001, 0.000001, 0.00001,0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
    'epsilon': [0.001, 0.01, 0.1, 1.0],
    'tol': [0.0001, 0.001, 0.01],
    'degree': [2, 3, 4, 5],
    'coef0' : [0,1,2,3],
    'max_iter': max_iter,
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'base_names': base_names,
}

config9 = {
    'C': np.linspace(0.001, 1.0, num=6).tolist() + list(range(2, 100,2)),# + list(range(1, 100,5)),# + list(range(100, 1000, 10))#np.logspace(-3, 3, 649)
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.0000001, 0.000001, 0.00001,0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
    'epsilon': [0.001, 0.01, 0.1, 1.0],
    'tol': [0.0001, 0.001, 0.01],
    'degree': [2, 3, 4, 5],
    'coef0' : [0,1,2,3],
    'max_iter': max_iter,
    'train_test_splits': train_test_splits,
    'repetitions':repetitions,
    'base_names': base_names,
}


# Now update your main function
def main():
    # Define your directories
    # Define the relative paths for data and results directories
    data_directory = os.path.join(current_dir, 'Singleoutput_SVR', 'datasets')
    results_directory = os.path.join(current_dir, 'Singleoutput_SVR', 'results')

    # Prompt user to select a configuration
    print("Select a configuration:")
    print("1: Configuration 1 (runSVR)")
    print("2: Configuration 2 (runSVR2)")
    print("3: Configuration 3 (runSVR3)")
    print("4: Configuration 4 (runSVR4)")
    print("5: Configuration 5 (runSVR5)")
    print("6: Configuration 6 (runSVR6)")
    print("7: Configuration 7 (runSVR7)")
    print("8: Configuration 8 (runSVR8)")
    print("9: Configuration 9 (runSVR9)")
    
    
    config_choice = input("Enter the configuration number (1-9): ")
    
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
    elif config_choice == '6':
        results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C6')
        config = config6
    elif config_choice == '7':
        results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C7')
        config = config7
    elif config_choice == '8':
        results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C8')
        config = config8
    elif config_choice == '9':
        results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C9')
        config = config9
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
                elif config_choice == '6':
                    print(f"Running Configuration 6 for: {base_name}")
                    runSVR_C6(data, base_name, results_directory, dataset_params[base_name], config)
                elif config_choice == '7':
                     print(f"Running Configuration 7 for: {base_name}")
                     runSVR_C7(data, base_name, results_directory, dataset_params[base_name], config)
                elif config_choice == '8':
                    print(f"Running Configuration 8 for: {base_name}")
                    runSVR_C8(data, base_name, results_directory, dataset_params[base_name], config)
                elif config_choice == '9':
                    print(f"Running Configuration 9 for: {base_name}")
                    runSVR_C9(data, base_name, results_directory, dataset_params[base_name], config)
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
                runSVR(data, user_choice, results_directory, dataset_params[base_name], config)#, dataset_params[user_choice]
            elif config_choice == '2':
                runSVR_C2(data, user_choice, results_directory, dataset_params[base_name], config)#, dataset_params[user_choice]
            elif config_choice == '3':
                runSVR_C3(data, user_choice, results_directory, dataset_params[base_name], config)#, dataset_params[user_choice]
            elif config_choice == '4':
                runSVR_C4(data, user_choice, results_directory, dataset_params[base_name], config)#, dataset_params[user_choice]
            elif config_choice == '5':
                runSVR_C5(data, user_choice, results_directory, dataset_params[base_name], config)#, dataset_params[user_choice]
            elif config_choice == '6':
                runSVR_C6(data, user_choice, results_directory, dataset_params[base_name], config)#, dataset_params[user_choice]
            elif config_choice == '7':
                runSVR_C7(data, user_choice, results_directory, dataset_params[base_name], config)#, dataset_params[user_choice]
            elif config_choice == '8':
                runSVR_C8(data, user_choice, results_directory, dataset_params[base_name], config)#, dataset_params[user_choice]
            elif config_choice == '9':
                runSVR_C9(data, user_choice, results_directory, dataset_params[base_name], config)#, dataset_params[user_choice]

if __name__ == "__main__":
    main()



##################################### in below code i am trying to handle the custome input for the dataset, but getting error , so it is commented


# def main():
#     # Define your directories
#     data_directory = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/dataset'
#     results_directory = '/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/lets_talk_about_svm'

#     # Prompt user to select a configuration
#     print("Select a configuration:")
#     print("1: Configuration 1 (runSVR)")
#     print("2: Configuration 2 (runSVR2)")
#     print("3: Configuration 3 (runSVR3)")
#     print("4: Configuration 4 (runSVR4)")
#     print("5: Configuration 5 (runSVR5)")
    
#     config_choice = input("Enter the configuration number (1-5): ")
    
#     # Set the results directory based on selected configuration
#     if config_choice == '1':
#         results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C1')
#         config = config1
#     elif config_choice == '2':
#         results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C2')
#         config = config2
#     elif config_choice == '3':
#         results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C3')
#         config = config3
#     elif config_choice == '4':
#         results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C4')
#         config = config4
#     elif config_choice == '5':
#         results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'svm_plain2024', 'C5')
#         config = config5
#     else:
#         print("Invalid choice. Exiting.")
#         return

#     # Make sure the results directory exists
#     os.makedirs(results_directory, exist_ok=True)

#     base_names = config['base_names']
#     # Process datasets
#     print("Starting dataset processing...")
#     processed_datasets, dataset_params = process_datasets(base_names, data_directory, results_directory)

#     while True:
#         print("\nAvailable datasets:")
#         for base_name in processed_datasets.keys():
#             print(f"- {base_name}")

#         print("Type 'all' to process all datasets or enter the name of a specific dataset (or 'q' to quit): ")
#         user_choice = input().strip()
        
#         if user_choice.lower() == 'q':
#             break

#         # Process all datasets if the user selects "all"
#         if user_choice.lower() == 'all':
#             for base_name in processed_datasets.keys():
#                 process_selected_dataset(config_choice, processed_datasets, base_name, results_directory, dataset_params, config)
#             print("Data processed successfully for all datasets.")
#             break
#         # Process a specific dataset
#         elif user_choice in processed_datasets:
#             process_selected_dataset(config_choice, processed_datasets, user_choice, results_directory, dataset_params, config)
#         else:
#             print("Invalid dataset name. Please try again.")

# def process_selected_dataset(config_choice, processed_datasets, base_name, results_directory, dataset_params, config):
#     data = processed_datasets[base_name]

#     print(f"Running SVR experiment for: {base_name}")
#     valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    
#     customize_params = input("Do you want to input custom parameters for this dataset? (y/n): ").lower()
#     if customize_params == 'y':
#         # Prompt for selective parameters
#         custom_C = input(f"Enter C value (default {config['C']}): ")
#         if custom_C:  # Only update if provided
#             try:
#                 config['C'] = float(custom_C)
#             except ValueError:
#                 print("Invalid input for C. Retaining default value.")

#         # Handle kernel input with default
#         custom_kernel = input(f"Select kernel type (default {config['kernel']}): ").lower().strip()
#         if custom_kernel in valid_kernels:
#             config['kernel'] = custom_kernel  # Update if valid
#         else:
#             print(f"Invalid kernel. Retaining default value: {config['kernel']}.")

#         custom_gamma = input(f"Enter gamma value (default {config['gamma']}): ")
#         if custom_gamma:  # Only update if provided
#             try:
#                 config['gamma'] = float(custom_gamma)
#             except ValueError:
#                 print("Invalid input for gamma. Retaining default value.")

#         custom_epsilon = input(f"Enter epsilon value (default {config['epsilon']}): ")
#         if custom_epsilon:  # Only update if provided
#             try:
#                 config['epsilon'] = float(custom_epsilon)
#             except ValueError:
#                 print("Invalid input for epsilon. Retaining default value.")

#     # Process the selected dataset with the configuration
#     if config_choice == '1':
#         runSVR(data, base_name, results_directory, dataset_params[base_name], config)
#     elif config_choice == '2':
#         runSVR_C2(data, base_name, results_directory, dataset_params[base_name], config)
#     elif config_choice == '3':
#         runSVR_C3(data, base_name, results_directory, dataset_params[base_name], config)
#     elif config_choice == '4':
#         runSVR_C4(data, base_name, results_directory, dataset_params[base_name], config)
#     elif config_choice == '5':
#         runSVR_C5(data, base_name, results_directory, dataset_params[base_name], config)

# if __name__ == "__main__":
#     main()
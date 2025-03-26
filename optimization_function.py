import csv
import os
import json
import pandas as pd



executed_configurations = set()
error_log = []
#this below log function will maintain the single log fill of all the kernels
def log_configuration_single_file(log_file, C, kernel_value, gamma, epsilon, degree=None, coef0=None, test_split=None, mean_error=None, base_name=None):
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # Create a list to hold the parameters
        params = [C, kernel_value, gamma, epsilon]
        # Add degree and coef0 only if they are provided (not None)
        if degree is not None:
            params.append(degree)
        if coef0 is not None:
            params.append(coef0)
        if test_split is not None:
            params.append(test_split)
        if mean_error is not None:
            params.append(mean_error)
        if base_name is not None:
            params.append(base_name)

        writer.writerow(params)

#calling as 
# log_configuration_single_file(log_file, C, kernel, gamma, epsilon, degree if 'degree' in config else None, coef0 if 'coef0' in config else None, test_size, mean_3d_error_testing, base_name)


# Load previously executed configurations from log file
# def load_executed_configurations(log_file):
#     if os.path.exists(log_file):
#         with open(log_file, 'r') as file:
#             return json.load(file)
#     return {}

def load_executed_configurations(log_file):
    executed_configs = set()  # Using a set for fast lookups
    try:
        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:  # Check if the row is not empty
                    executed_configs.add(tuple(row))  # Add as a tuple to the set
    except Exception as e:
        print(f"Error reading the log file {log_file}: {e}")
    return executed_configs

# def load_executed_configurations(log_file):
#     # Check if file exists
#     if not os.path.exists(log_file):
#         print(f"Log file '{log_file}' does not exist.")
#         return {}

#     configurations = []
#     try:
#         with open(log_file, 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if row:  # Check if the row is not empty
#                     configurations.append(row)  # Append the valid row
#     except Exception as e:
#         print(f"Error reading the log file {log_file}: {e}")

#     return configurations

# Save executed configuration to log file
def save_executed_configurations(log_file, executed_configurations):
    with open(log_file, 'w') as file:
        json.dump(executed_configurations, file, indent=4)

def log_configuration(log_file, C, kernel_value, gamma, epsilon, degree, coef0, test_split, mean_error, dataset_name):
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dataset_name, C, kernel_value, gamma, epsilon, degree, coef0, test_split, mean_error])  # Include dataset_name

# Check if this configuration has already been executed
def is_executed(config, executed_configurations):
    config_str = json.dumps(config, sort_keys=True)  # Convert dict to string for comparison
    return config_str in executed_configurations

# Mark this configuration as executed
def mark_executed(config, executed_configurations, val_error, test_error):
    config_str = json.dumps(config, sort_keys=True)
    executed_configurations[config_str] = {
        "val_error": val_error,
        "test_error": test_error
    }

def update_executed_configurations(log_file, config):
    # Load existing configurations
    executed_configurations = load_executed_configurations(log_file)
    
    # Add the new configuration
    executed_configurations.append(config)
    
    # Save the updated list back to the log file
    with open(log_file, 'w') as file:
        json.dump(executed_configurations, file)


def is_configuration_executed(log_file, base_name, config):
    # Check if the log file exists
    if not os.path.exists(log_file):
        print(f"Log file {log_file} does not exist.")
        return False  # If the log file does not exist, return False

    # Load the log file into a DataFrame
    try:
        log_df = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return False

    # Print the columns and a few rows for debugging
    #print("Columns in log DataFrame:", log_df.columns)
    #print("First few rows in log DataFrame:", log_df.head())

    # Check if the required columns exist in the log file
    required_columns = ['Dataset', 'C', 'kernel', 'gamma', 'epsilon', 'degree', 'coef0']
    missing_columns = [col for col in required_columns if col not in log_df.columns]

    if missing_columns:
        print(f"Missing columns in the log file: {missing_columns}")
        return False

    # Create a condition to check if the specific configuration already exists
    condition = (
        (log_df['Dataset'] == base_name) &
        (log_df['C'] == config.get('C')) &
        (log_df['kernel'] == config.get('kernel')) &
        (log_df['gamma'] == config.get('gamma')) &
        (log_df['epsilon'] == config.get('epsilon')) &
        (log_df['degree'] == config.get('degree', None)) &
        (log_df['coef0'] == config.get('coef0', None))
    )

    # Check if any rows match the condition
    config_exists = log_df[condition].shape[0] > 0

    if config_exists:
        print(f"Configuration for dataset '{base_name}' with these parameters already exists in the log.")
    else:
        print(f"Configuration for dataset '{base_name}' with these parameters does not exist in the log.")

    return config_exists


#both are working before above, revert if above same not work
def is_executedc1(config, base_name):
    # Convert lists within the config to tuples to make them hashable
    config_tuple = tuple((key, tuple(value) if isinstance(value, list) else value) for key, value in config.items())
    # Include the base_name to make the configuration unique to the dataset
    return (base_name, config_tuple) in executed_configurations

def mark_executedc1(config, base_name):
    # Convert any lists in config to tuples to make config_tuple hashable
    config_tuple = tuple((key, tuple(value) if isinstance(value, list) else value) for key, value in config.items())
    # Add the hashable config_tuple and base_name to the set
    executed_configurations.add((base_name, config_tuple))


def save_error_log(config, base_name, mean_error_val, mean_error_testing):
    error_log.append({
        'config': config,
        'dataset': base_name,
        'validation_error': mean_error_val,
        'testing_error': mean_error_testing
    })

def compare_errors():
    """Compare all errors and find the configuration with the minimum error."""
    min_error_config = min(error_log, key=lambda x: x['testing_error'])
    print(f"Best configuration so far: {min_error_config['config']}")
    print(f"Best testing error: {min_error_config['testing_error']:.5f} m")
    

#this log function will creat individual log file for each kernel
def log_configuration_individual_file(log_file, C, kernel_value, gamma, epsilon, degree=None, coef0=None, test_split=None, mean_error=None, base_name=None):
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write the parameters as before
        writer.writerow([C, kernel_value, gamma, epsilon, degree, coef0, test_split, mean_error, base_name])

def get_log_file(kernel, results_directory):
    return os.path.join(results_directory, f'executed_configs_{kernel}_log.csv')

# In your runSVR function, call it like this
# log_file = get_log_file(kernel, results_directory)
# log_configuration(log_file, C, kernel, gamma, epsilon, degree, coef0, test_size, mean_3d_error_testing, base_name)

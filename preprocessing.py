import os
import copy
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
#from configurations_functions_svm import (remapBldDB, remapFloorDB, replace_non_detected_values_orig,data_rep_positive)
import warnings
from sklearn.exceptions import ConvergenceWarning

def remapFloorDB(database, origFloors, newFloors):
    mapping = dict(zip(origFloors, newFloors))
    for key in ['trncrd', 'tstcrd']:
        database[key][:, 3] = np.array([mapping.get(floor, floor) for floor in database[key][:, 3]])
    return database

def remapBldDB(database, origBlds, newBlds):
    mapping = dict(zip(origBlds, newBlds))
    for key in ['trncrd', 'tstcrd']:
        database[key][:, 4] = np.array([mapping.get(bld, bld) for bld in database[key][:, 4]])
    return database

def minmaxscaler(X_train, X_val, X_testing):
    """Scales features using MinMaxScaler."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_testing_scaled = scaler.transform(X_testing)
    return X_train_scaled, X_val_scaled, X_testing_scaled

def clear_terminal():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def ensure_directory_exists(directory):
    """Creates the directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Handling data representation: convert data to positive RSSI values
def data_rep_positive(database):
    min_rssi = min(database['trnrss'].min(), database['tstrss'].min())
    database['trnrss'] -= min_rssi
    database['tstrss'] -= min_rssi
    
    # Copy the labels (assuming they exist in the same structure)
    database['trncrd'] = database.get('trncrd', None)
    database['tstcrd'] = database.get('tstcrd', None)
    
    return database


def process_datasets(base_names, data_directory, results_directory,):
    """Processes datasets and returns a structured database."""
    
    # Set seeds for reproducibility
    # random.seed(420)
    # np.random.seed(420)
    # tf.random.set_seed(420)

    clear_terminal()
    warnings.filterwarnings("error", category=ConvergenceWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    ensure_directory_exists(results_directory)

    # Initialize a database to hold all processed data
    processed_datasets = {}
    dataset_params = {}

    for base_name in base_names:
        print(f"Processing dataset: {base_name}")

        # Construct file paths
        train_coord_file = os.path.join(data_directory, f"{base_name}_trncrd.csv")
        train_rssi_file = os.path.join(data_directory, f"{base_name}_trnrss.csv")
        test_coord_file = os.path.join(data_directory, f"{base_name}_tstcrd.csv")
        test_rssi_file = os.path.join(data_directory, f"{base_name}_tstrss.csv")

        # Check if all required files exist
        if not all(os.path.exists(file) for file in [train_coord_file, train_rssi_file, test_coord_file, test_rssi_file]):
            print(f"Missing files for {base_name}, skipping...")
            continue

        # Load coordinate and RSSI data
        coord_columns = ['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']
        train_df_coord = pd.read_csv(train_coord_file, header=None, names=coord_columns)
        test_df_coord = pd.read_csv(test_coord_file, header=None, names=coord_columns)
        train_df_rssi = pd.read_csv(train_rssi_file, header=None)
        test_df_rssi = pd.read_csv(test_rssi_file, header=None)

        # Create a dictionary for the database
        database_main = {
            'trncrd': train_df_coord.values,
            'tstcrd': test_df_coord.values,
            'trnrss': train_df_rssi.values,
            'tstrss': test_df_rssi.values
        }
        
        # Create an independent deep copy of the original database
        database_orig = copy.deepcopy(database_main)
        
        # Remap building and floor IDs
        origBlds = np.unique(database_orig['trncrd'][:, 4])
        nblds = len(origBlds)
        database0 = remapBldDB(database_orig, origBlds, np.arange(1, nblds + 1))
        
        origFloors = np.unique(database_orig['trncrd'][:, 3])
        nfloors = len(origFloors)
        database0 = remapFloorDB(database0, origFloors, np.arange(1, nfloors + 1))
        
        # Define non-detected values
        defaultNonDetectedValue = 100
        newNonDetectedValue = [] 
        
        
        # Handle non-detected RSSI values
        minValueDetected = min(np.min(database0['trnrss']), np.min(database0['tstrss']))
            
        if len(newNonDetectedValue) == 0:
            newNonDetectedValue = minValueDetected -1
        
        #Manual fix for WGS84 datasets and UEXBx datasets
        if np.min(database0['trnrss']) == -200:
            defaultNonDetectedValue = -200
            newNonDetectedValue = -200

        if np.min(database0['trnrss']) == -110 and np.max(database0['trnrss']) < 0:
            idxT = database0['trnrss'] <= -109
            idxV = database0['tstrss'] <= -109

            database_orig['trnrss'][idxT] = -110
            database_orig['tstrss'][idxV] = -110

            database0['trnrss'][idxT] = -110
            database0['tstrss'][idxV] = -110

            defaultNonDetectedValue = -110
            newNonDetectedValue = -110

        if np.min(database0['trnrss']) == -109 and np.max(database0['trnrss']) < 0:
            idxT = database0['trnrss'] <= -108
            idxV = database0['tstrss'] <= -108

            database_orig['trnrss'][idxT] = -109
            database_orig['tstrss'][idxV] = -109

            database0['trnrss'][idxT] = -109
            database0['tstrss'][idxV] = -109

            defaultNonDetectedValue = -109
            newNonDetectedValue = -109
            
        #Handling Non detected values
        if defaultNonDetectedValue != 0: 
            database0 = replace_non_detected_values_orig(database0, defaultNonDetectedValue, newNonDetectedValue)
        database = data_rep_positive(database0)
            
        rsamples1 = database['trnrss'].shape[0]
        osamples1 = database['tstrss'].shape[0]
        nmacs1 = database['tstrss'].shape[1]
        
        # Create boolean arrays to indicate valid Mac addresses (APs)
        database['trainingValidMacs'] = (database_main['trnrss'] != defaultNonDetectedValue) 
        database['testValidMacs'] = (database_main['tstrss'] != defaultNonDetectedValue)
        
        # Count and get the total number of valid access points in both training and test data
        trainingValidMacs = (database['trainingValidMacs'].sum(axis=0) > 0).sum()
        testValidMacs = (database['testValidMacs'].sum(axis=0) > 0).sum()
    
        print(f'Total valid access points in training data: {trainingValidMacs}')
        print(f'Total valid access points in test data: {testValidMacs}')

        vecidxmacs = np.arange(nmacs1)
        vecidxTsamples = np.arange(rsamples1)
        vecidxVsamples = np.arange(osamples1)
        
        validMacs = vecidxmacs[np.sum(database['trainingValidMacs'], axis=0) > 0] 

        # Keep only the valid Mac addresses
        database['trnrss'] = database['trnrss'][:, validMacs]
        database['trainingValidMacs'] = database['trainingValidMacs'][:, validMacs]
        database['tstrss'] = database['tstrss'][:, validMacs]
        database['testValidMacs'] = database['testValidMacs'][:, validMacs]

        # Clean void fingerprints
        validTSamples = vecidxTsamples[np.sum(database['trainingValidMacs'], axis=1) > 0]
        #validTSamples = vecidxTsamples[np.sum(database['trainingValidMacs'] & (database['trnrss_missing'] == 0), axis=1) > 0]

        database['trnrss'] = database['trnrss'][validTSamples, :]
        database['trainingValidMacs'] = database['trainingValidMacs'][validTSamples, :]
        database['trncrd'] = database['trncrd'][validTSamples, :]

        validVSamples = vecidxVsamples[np.sum(database['testValidMacs'], axis=1) > 0]
        #validVSamples = vecidxVsamples[np.sum(database['testValidMacs'] & (database['tstrss_missing'] == 0), axis=1) > 0]
        
        database['tstrss'] = database['tstrss'][validVSamples, :]
        database['testValidMacs'] = database['testValidMacs'][validVSamples, :]
        database['tstcrd'] = database['tstcrd'][validVSamples, :]
              
        # Check shapes for consistency
        assert database['trnrss'].shape[0] == database['trncrd'].shape[0], "Mismatch in training RSSI and coordinate sample sizes"
        assert database['tstrss'].shape[0] == database['tstcrd'].shape[0], "Mismatch in test RSSI and coordinate sample sizes"

        rsamples = database['trnrss'].shape[0]
        osamples = database['tstrss'].shape[0]
        nmacs = database['tstrss'].shape[1]
        
          # Pack parameters into a dictionary
        params = {
            'rsamples1': rsamples1,
            'osamples1': osamples1,
            'nmacs1': nmacs1,
            'rsamples': rsamples,
            'osamples': osamples,
            'nmacs': nmacs,
            'minValueDetected': minValueDetected,
            'defaultNonDetectedValue': defaultNonDetectedValue,
            'newNonDetectedValue': newNonDetectedValue
        }
        
        #print(f"Type of processed_datasets before iteration: {type(processed_datasets)}")
    
        # Save the processed dataset into the dictionary
        processed_datasets[base_name] = database
        # Append parameters for this dataset to the list
        dataset_params[base_name] = params
        
        
    return processed_datasets, dataset_params#,database_orig


# Define the function to replace old null values with new null values
def datarepNewNull(arr, old_null, new_null):
    """
    Replace old_null with new_null in the given array.
    """
    return np.where(arr == old_null, new_null, arr)

def datarepNewNullDB(db0, old_null, new_null):
    """
    Replace old_null with new_null in the trainingMacs and testMacs fields of the database.
    """
    db1 = {}
    db1['trnrss'] = datarepNewNull(db0['trnrss'], old_null, new_null)
    db1['tstrss'] = datarepNewNull(db0['tstrss'], old_null, new_null)
    db1['trncrd'] = db0['trncrd']
    db1['tstcrd'] = db0['tstcrd']
    return db1

def datarep_new_null(m0, old_null, new_null):
    """
    Replace old null values with new specified values in the matrix m0.
    
    Args:
        m0 (np.ndarray): The input matrix.
        old_null (float or int): The value to be replaced.
        new_null (float or int): The value to replace old_null with.

    Returns:
        np.ndarray: The matrix with old_null values replaced by new_null.
    """
    # Perform the replacement
    m1 = m0 * (m0 != old_null) + new_null * (m0 == old_null)
    return m1

def replace_non_detected_values_orig(database, old_null, new_null):
    """
    Replace old non-detected RSSI values with new specified values in both training and test datasets.
    
    Args:
        database (dict): A dictionary containing 'trnrss', 'tstrss', 'trncrd', and 'tstcrd'.
        old_null (float or int): The old null value to be replaced (e.g., -109).
        new_null (float or int): The new value to replace the old null value with.

    Returns:
        dict: Updated database with replaced RSSI values.
        tuple: Boolean masks showing where values were replaced.
    """
     # Create masks to track where non-detected values are
    trnrss_mask = (database['trnrss'] == old_null)
    tstrss_mask = (database['tstrss'] == old_null)

    # Store the initial values before replacement
    initial_trnrss_values = database['trnrss'][trnrss_mask]
    initial_tstrss_values = database['tstrss'][tstrss_mask]
    
    
    # Replace in training RSSI data using datarep_new_null
    database['trnrss'] = datarep_new_null(database['trnrss'], old_null, new_null)

    # Replace in test RSSI data using datarep_new_null
    database['tstrss'] = datarep_new_null(database['tstrss'], old_null, new_null)

    # Assuming the labels (coordinates) remain unchanged
    database['trncrd'] = database.get('trncrd', None)
    database['tstcrd'] = database.get('tstcrd', None)

    return database#, initial_trnrss_values, initial_tstrss_values, trnrss_mask, tstrss_mask



def calculate_height_difference(test_labels, predicted_point):
    """
    Calculate the height difference.

    Parameters:
    - test_labels: The true labels (altitude).
    - predicted_point: The predicted labels (altitude).

    Returns:
    - Height difference.
    """
    return abs(test_labels[2] - predicted_point[2])

def calculate_floor_difference(predicted_floor, true_floor):
    """
    Calculate the floor difference.

    Parameters:
    - predicted_floor: The predicted floor.
    - true_floor: The true floor.

    Returns:
    - Floor difference.
    """
    return abs(predicted_floor - true_floor)

def calculate_building_error(predicted_building, true_building):
    """
    Calculate the building error.

    Parameters:
    - predicted_building: The predicted building.
    - true_building: The true building.

    Returns:
    - Building error (0 if the same, 1 if different).
    """
    return int(predicted_building != true_building)

def data_rep_positive_old(database, distance_metrics, newNonDetectedValue):
    # Convert RSSI values to positive by subtracting the minimum value
    min_rssi = min(database['trnrss'].min(), database['tstrss'].min())
    database['trnrss'] -= min_rssi
    database['tstrss'] -= min_rssi

    # Copy the labels (assuming they exist in the same structure)
    database['trncrd'] = database.get('trncrd', None)
    database['tstcrd'] = database.get('tstcrd', None)
    
    # Handle additional parameters based on distanceMetric
    if 'PLGD' in distance_metrics:
        additionalparams = -85 - newNonDetectedValue
    else:
        additionalparams = 0

    return database, additionalparams

def remap_vector(m0, vin, vout):
    # Create an output array with the same shape as m0, initialized to 0
    m1 = np.zeros_like(m0)

    # Iterate through the elements of vin and vout, remapping values in m0 to vout
    for i in range(len(vin)):
        m1[m0 == vin[i]] = vout[i]

    return m1




   
def datarep_powed(database):
    """
    Applies the powered data representation transformation to the RSSI values in the database.

    Parameters:
        database (dict): A dictionary containing 'trnrss' (training RSSI values) and 'tstrss' (test RSSI values).
    
    Returns:
        transformed_db (dict): A dictionary containing the transformed 'trnrss' and 'tstrss' values, and the original labels.
    """
    trainingMacs = database['trnrss']
    testMacs = database['tstrss']
    
    # Calculate minimum value
    minValue = np.min([trainingMacs.min(), testMacs.min()])
    
    # Normalize value
    normValue = (-minValue) ** np.exp(1)
    
    # Apply transformation
    transformed_trainingMacs = ((trainingMacs - minValue) ** np.exp(1)) / normValue
    transformed_testMacs = ((testMacs - minValue) ** np.exp(1)) / normValue
    
    # Create the transformed database
    transformed_db = {
        'trnrss': transformed_trainingMacs,
        'tstrss': transformed_testMacs,
        'trainingLabels': database.get('trainingLabels', None),
        'testLabels': database.get('testLabels', None)
    }
    
    return transformed_db

def datarepExponential(db0):
    # Calculate the minimum value from both trainingMacs and testMacs
    minValue = np.min(np.concatenate((db0.trainingMacs.flatten(), db0.testMacs.flatten())))
    
    # Calculate the normalization factor
    normValue = np.exp(-minValue / 24)
    
    # Perform the exponential transformation and normalization
    transformed_trainingMacs = np.exp((db0.trainingMacs - minValue) / 24) / normValue
    transformed_testMacs = np.exp((db0.testMacs - minValue) / 24) / normValue
    Data=None
    # Create a new instance of Data for the transformed values
    db1 = Data(
        trainingMacs=transformed_trainingMacs,
        testMacs=transformed_testMacs,
        trainingLabels=db0.trainingLabels,
        testLabels=db0.testLabels
    )
    
    return db1

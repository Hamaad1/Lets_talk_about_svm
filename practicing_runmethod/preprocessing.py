import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from configurations_functions_svm import (remapBldDB, remapFloorDB, replace_non_detected_values_orig)
import logging
import warnings
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, train_test_split, cross_val_score

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


    
def process_datasets(base_names, data_directory, results_directory,):
    """Processes datasets and returns a structured database."""
    
    # Set seeds for reproducibility
    random.seed(420)
    np.random.seed(420)
    tf.random.set_seed(420)

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
        nan_value = np.nan 
        
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
            database = replace_non_detected_values_orig(database0, defaultNonDetectedValue, newNonDetectedValue)
        #database = data_rep_positive(database0)
            
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
        
        
    return processed_datasets, dataset_params  


# def load_data(dataset_name, data_directory):
#     """Loads data based on the dataset name."""
#     train_rssi_file = os.path.join(data_directory, f"{dataset_name}_trnrss.csv")
#     test_rssi_file = os.path.join(data_directory, f"{dataset_name}_tstrss.csv")
#     train_coords_file = os.path.join(data_directory, f"{dataset_name}_trncrd.csv")
#     test_coords_file = os.path.join(data_directory, f"{dataset_name}_tstcrd.csv")
    
#     # Load the data
#     train_rssi_data = pd.read_csv(train_rssi_file)
#     test_rssi_data = pd.read_csv(test_rssi_file)
#     train_coords_data = pd.read_csv(train_coords_file)
#     test_coords_data = pd.read_csv(test_coords_file)
    
#     return {
#         'trnrss': train_rssi_data.values, 
#         'tstrss': test_rssi_data.values, 
#         'trncrd': train_coords_data.values, 
#         'tstcrd': test_coords_data.values
#     }
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.svm import OneClassSVM
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import pairwise_distances



# Define a similarity function (example: cosine similarity)
def my_similarity_function(test_sample, train_samples):
    test_norm = np.linalg.norm(test_sample)
    train_norm = np.linalg.norm(train_samples, axis=1, keepdims=True)
    dot_product = np.dot(train_samples, test_sample)
    similarity_values = dot_product / (train_norm * test_norm)
    return similarity_values

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

# Define the distance metric function (if not predefined)
def compute_distances(trainingMacs, ofp, distance_metric,additionalparams):# alpha=None
    if distance_metric == 'cityblock':
        return cdist(trainingMacs, ofp.reshape(1, -1), metric='cityblock').flatten()
    elif distance_metric == 'euclidean':
        return cdist(trainingMacs, ofp.reshape(1, -1), metric='euclidean').flatten()
    elif distance_metric == 'minkowski3':
        return cdist(trainingMacs, ofp.reshape(1, -1), metric='minkowski', p=3).flatten()
        
    elif distance_metric == 'sorensen':
        # Sørensen distance implementation
        return np.sum(np.abs(trainingMacs - ofp), axis=1) / np.sum(trainingMacs + ofp, axis=1)
    # Add more metrics as needed
    elif distance_metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        dot_products = np.sum(trainingMacs * ofp, axis=1)
        training_magnitudes = np.linalg.norm(trainingMacs, axis=1)
        ofp_magnitude = np.linalg.norm(ofp)
        cosine_similarity = dot_products / (training_magnitudes * ofp_magnitude)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)  # Ensure values are within valid range
        return 1 - cosine_similarity
    elif distance_metric == 'neyman':
        factorzero = 0.000001 if np.max(trainingMacs) <= 1.0001 and np.max(ofp) <= 1.0001 else 0.0001
        divisor = ofp + (factorzero * (ofp == 0))
        return np.sum(((trainingMacs - ofp) ** 2) / divisor, axis=1)
    elif distance_metric == 'neyman2':
        factorzero = 0.000001 if np.max(trainingMacs) <= 1.0001 and np.max(ofp) <= 1.0001 else 0.0001
        divisor = trainingMacs + (factorzero * (trainingMacs == 0))
        return np.sum(((trainingMacs - ofp) ** 2) / divisor, axis=1)
    elif distance_metric == 'lgd':
        sigma = 5
        threshold = 0.0001
        numerator = -((trainingMacs - ofp) ** 2)
        denominator = 2 * (sigma ** 2)
        differences = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(numerator / denominator)
        nonzero = (trainingMacs * ofp) != 0
        return -np.sum(np.log(differences * nonzero + threshold * (1 - nonzero)), axis=1)
    elif distance_metric == 'plgd10':
        # Assuming 'alpha' is used as the threshold
        threshold = additionalparams
        p1 = np.sum((trainingMacs - threshold) * (trainingMacs >= threshold) * (ofp == 0), axis=1)
        p2 = np.sum((ofp - threshold) * (ofp >= threshold) * (trainingMacs == 0), axis=1)
        d_lgd = compute_distances(trainingMacs, ofp, distance_metric='lgd', additionalparams=additionalparams)
        return d_lgd + (1 / 10) * (p1 + p2)
    elif distance_metric == 'plgd40':
        threshold = additionalparams
        p1 = np.sum((trainingMacs - threshold) * (trainingMacs >= threshold) * (ofp == 0), axis=1)
        p2 = np.sum((ofp - threshold) * (ofp >= threshold) * (trainingMacs == 0), axis=1)
        d_lgd = compute_distances(trainingMacs, ofp, distance_metric='lgd', additionalparams=additionalparams)
        return d_lgd + (1 / 40) * (p1 + p2)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")




# Function to train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test_combined, C=1.0, kernel='rbf', gamma='scale'):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SVR model and MultiOutputRegressor
    svr = SVR(kernel=kernel, C=C, gamma=gamma)
    multi_output_svr = MultiOutputRegressor(svr)

    # Train the SVR model for multi-output prediction (latitude, longitude, altitude)
    multi_output_svr.fit(X_train_scaled, y_train)

    # Make predictions for test data
    y_pred_combined = multi_output_svr.predict(X_test_scaled)

    # Calculate 3D error (Euclidean distance)
    #errors = cdist(y_pred_combined, y_pred_combined.reshape(1, -1), metric='euclidean').flatten()
    errors = pairwise_distances(y_pred_combined, y_pred_combined, metric='euclidean')
    mean_3d_error = np.mean(np.diag(errors))

    # Calculate Mean Squared Errors for each target
    mse_lat = mean_squared_error(y_test_combined[:, 0], y_pred_combined[:, 0])
    mse_lon = mean_squared_error(y_test_combined[:, 1], y_pred_combined[:, 1])
    mse_alt = mean_squared_error(y_test_combined[:, 2], y_pred_combined[:, 2])

    return mean_3d_error, mse_lat, mse_lon, mse_alt

def compute_weighted_centroid(positions, distances, strategy='unweighted'):
    epsilon = 1e-8
    if strategy == 'unweighted':
        weights = np.ones(len(distances))
    elif strategy == 'inverse_distance':
        weights = 1 / (distances + epsilon)
    elif strategy == 'squared_inverse_distance':
        weights = 1 / (distances ** 2 + epsilon)
    else:
        raise ValueError("Unknown strategy.")
    weights /= weights.sum()
    weighted_positions = np.dot(weights, positions)
    return weighted_positions


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





def replace_non_detected_values_nan(database, old_null):
    """
    Replace old non-detected RSSI values with NaN in both training and test datasets.
    
    Args:
        database (dict): A dictionary containing 'trnrss', 'tstrss', 'trncrd', and 'tstcrd'.
        old_null (float or int): The old null value to be replaced (e.g., -109).

    Returns:
        dict: Updated database with replaced RSSI values.
    """
        # Convert RSSI arrays to float to allow NaN values
    database['trnrss'] = database['trnrss'].astype(float)
    database['tstrss'] = database['tstrss'].astype(float)
    
    # Create masks to track where non-detected values are
    trnrss_mask = (database['trnrss'] == old_null)
    tstrss_mask = (database['tstrss'] == old_null)

    # Replace non-detected values with NaN in training RSSI data
    database['trnrss'][trnrss_mask] = np.nan

    # Replace non-detected values with NaN in test RSSI data
    database['tstrss'][tstrss_mask] = np.nan

    # The coordinates (trncrd and tstcrd) remain unchanged
    database['trncrd'] = database.get('trncrd', None)
    database['tstcrd'] = database.get('tstcrd', None)
    
    # Create missing value indicators (optional but useful)
    train_missing_indicator = trnrss_mask.astype(int)
    test_missing_indicator = tstrss_mask.astype(int)

    # Add missing indicator columns
    database['trnrss_missing'] = train_missing_indicator
    database['tstrss_missing'] = test_missing_indicator

    return database

def replace_and_flag_non_detected_values_100_Nan_0(database, old_null):
    """
    Replace old non-detected RSSI values with NaN, then replace NaN with 0 while flagging 0 as non-detected.
    
    Args:
        database (dict): A dictionary containing 'trnrss', 'tstrss', 'trncrd', and 'tstcrd'.
        old_null (float or int): The old non-detected value to be replaced (e.g., 100).
        
    Returns:
        dict: Updated database with RSSI values replaced and flagged.
    """
    # Convert RSSI arrays to float to allow NaN values
    database['trnrss'] = database['trnrss'].astype(float)
    database['tstrss'] = database['tstrss'].astype(float)
    
    # Step 1: Replace old non-detected values (like 100) with NaN
    trnrss_mask = (database['trnrss'] == old_null)
    tstrss_mask = (database['tstrss'] == old_null)
    database['trnrss'][trnrss_mask] = np.nan
    database['tstrss'][tstrss_mask] = np.nan

    # Step 2: Replace NaN with 0 while creating a flag for these non-detected values
    trnrss_flag_mask = np.isnan(database['trnrss'])
    tstrss_flag_mask = np.isnan(database['tstrss'])
    database['trnrss'][trnrss_flag_mask] = 0
    database['tstrss'][tstrss_flag_mask] = 0
    
    # Step 3: Preserve original negative values and create missing value indicators
    train_missing_indicator = trnrss_flag_mask.astype(int)
    test_missing_indicator = tstrss_flag_mask.astype(int)

    # Add missing indicator columns (flagged as non-detected)
    database['trnrss_missing'] = train_missing_indicator
    database['tstrss_missing'] = test_missing_indicator

    return database

def replace_and_flag_non_detected_values_100_to_0(database, old_null):
    """
    Directly replace old non-detected RSSI values with 0, and flag those values as non-detected.
    
    Args:
        database (dict): A dictionary containing 'trnrss', 'tstrss', 'trncrd', and 'tstcrd'.
        old_null (float or int): The old non-detected value to be replaced (e.g., 100).
        
    Returns:
        dict: Updated database with RSSI values replaced and flagged.
    """
    # Convert RSSI arrays to float to ensure consistent processing
    database['trnrss'] = database['trnrss'].astype(float)
    database['tstrss'] = database['tstrss'].astype(float)
    
    # Step 1: Create a mask to identify non-detected values (e.g., 100)
    trnrss_mask = (database['trnrss'] == old_null)
    tstrss_mask = (database['tstrss'] == old_null)
    
    # Step 2: Directly replace non-detected values (e.g., 100) with 0
    database['trnrss'][trnrss_mask] = 0
    database['tstrss'][tstrss_mask] = 0
    
    # Step 3: Flag the non-detected values in 'trnrss_missing' and 'tstrss_missing'
    # train_missing_indicator = trnrss_mask.astype(int)
    # test_missing_indicator = tstrss_mask.astype(int)
    train_missing_indicator = (database['trnrss'] == 0).astype(int)
    test_missing_indicator = (database['tstrss'] == 0).astype(int)

    # Add missing indicator columns (flagged as non-detected)
    database['trnrss_missing'] = train_missing_indicator
    database['tstrss_missing'] = test_missing_indicator
    

    return database


def remove_outliers(X):
    # Initialize the One-Class SVM model
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.5)  # Adjust nu as needed
    
    # Fit the model to the data
    model.fit(X)
    
    # Predict outliers
    y_pred = model.predict(X)  # -1 for outliers, 1 for inliers
    
    # Keep only inliers (1)
    return X[y_pred == 1]


def standard_scaler_new(X_train, X_test, X_testing):# X_test,
    # Convert to numpy arrays if not already
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    X_testing_np = X_testing.values if hasattr(X_testing, 'values') else X_testing
    
    # Create masks for non-zero (detected) values
    train_mask = (X_train_np != 0)
    test_mask = (X_test_np != 0)
    testing_mask = (X_testing_np != 0)

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit scaler only on the detected values (non-zero) from the training set
    detected_train_values = X_train_np[train_mask]
    scaler.fit(detected_train_values.reshape(-1, 1))

    # Transform the training, test, and validation datasets
    X_train_scaled = np.where(train_mask, scaler.transform(X_train_np.reshape(-1, 1)).reshape(X_train_np.shape), X_train_np)
    X_test_scaled = np.where(test_mask, scaler.transform(X_test_np.reshape(-1, 1)).reshape(X_test_np.shape), X_test_np)
    X_testing_scaled = np.where(testing_mask, scaler.transform(X_testing_np.reshape(-1, 1)).reshape(X_testing_np.shape), X_testing_np)

    # Plotting
    plt.figure(figsize=(12, 6))
    i=2
    # Original training data
    plt.subplot(1, 3, 1)
    plt.title("Original Training Data")
    plt.scatter(range(len(X_train_np[:, i])), X_train_np[:, i], color='blue', label='Detected Values')
    plt.scatter(range(len(X_train_np[:, i])), X_train_np[:, i] * ~train_mask[:, i], color='red', label='Zero Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()

    # Scaled training data
    plt.subplot(1, 3, 2)
    plt.title("Scaled Training Data")
    plt.scatter(range(len(X_train_scaled[:, i])), X_train_scaled[:, i], color='green', label='Scaled Detected Values')
    plt.scatter(range(len(X_train_scaled[:, i])), X_train_scaled[:, i] * ~train_mask[:, i], color='red', label='Zero Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()

    # Skewness Plot (Histogram of detected values)
    plt.subplot(1, 3, 3)
    plt.title("Skewness of Detected Values (Original Data)")
    plt.hist(detected_train_values, bins=30, color='purple', alpha=0.7)
    plt.axvline(np.mean(detected_train_values), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(detected_train_values):.2f}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    #plt.show()

    # Calculate skewness
    skewness = skew(detected_train_values)
    print(f"Skewness of the detected values: {skewness:.2f}")
    
    return X_train_scaled,X_test_scaled, X_testing_scaled # X_test_scaled,

def standard_scaler_new0(X_train, X_val=None, X_test=None):
    """
    Scales the input training, validation, and test data using StandardScaler,
    while handling non-detected values.
    
    Args:
        X_train (pd.Series or np.ndarray): Training data with potential non-detected values.
        X_val (pd.Series or np.ndarray, optional): Validation data with potential non-detected values.
        X_test (pd.Series or np.ndarray, optional): Test data with potential non-detected values.
        
    Returns:
        tuple: Scaled training, validation, and test datasets.
    """
    # Convert RSSI arrays to float
    X_train = X_train.astype(float)
    if X_val is not None:
        X_val = X_val.astype(float)
    if X_test is not None:
        X_test = X_test.astype(float)

    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Step 1: Fit scaler on training data while ignoring NaNs
    train_rssi_values = X_train[~np.isnan(X_train)]
    
    if train_rssi_values.size > 0:
        scaler.fit(train_rssi_values.values.reshape(-1, 1))
        
        # Step 2: Transform the training data
        X_train_scaled = np.where(
            np.isnan(X_train),
            np.nan,  # Retain NaNs for non-detected values
            scaler.transform(X_train.values.reshape(-1, 1)).flatten()
        )
    else:
        X_train_scaled = np.full(X_train.shape, np.nan)

    # Step 3: Transform validation data if provided
    if X_val is not None:
        X_val_scaled = np.where(
            np.isnan(X_val),
            np.nan,  # Retain NaNs for non-detected values
            scaler.transform(X_val.values.reshape(-1, 1)).flatten()
        )
    else:
        X_val_scaled = None

    # Step 4: Transform test data if provided
    if X_test is not None:
        X_test_scaled = np.where(
            np.isnan(X_test),
            np.nan,  # Retain NaNs for non-detected values
            scaler.transform(X_test.values.reshape(-1, 1)).flatten()
        )
    else:
        X_test_scaled = None
    
    return X_train_scaled, X_val_scaled, X_test_scaled


def minmaxscaler(X_train, X_val=None, X_test=None):
    """
    Scales the input training, validation (if present), and test data using StandardScaler.
    """
    scaler = MinMaxScaler()  # range can be feature_range=(0, 1)(-1,1)

    # Fit scaler on training data
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
    # Print the min and max values of the original data used by the scaler
    # print(f"Data Min: {scaler.data_min_}")
    # print(f"Data Max: {scaler.data_max_}")

    # Additionally, you can print the scaled range if needed
    print(f"Scaled Data Range: {X_train_scaled.min()} to {X_train_scaled.max()}")

    # Scale validation data if provided
    if X_val is not None and len(X_val) > 0:
        X_val_scaled = scaler.transform(X_val)  # Only transform validation data
    else:
        X_val_scaled = None  # No validation data to scale

    # Scale test data if provided
    if X_test is not None and len(X_test) > 0:
        X_testing_scaled = scaler.transform(X_test)  # Only transform test data
    else:
        X_testing_scaled = None  # No test data to scale

    return X_train_scaled, X_val_scaled, X_testing_scaled

def standard_scaler_official(X_train, X_val, X_testing):
    """
    Apply StandardScaler to X_train, X_val, and X_testing.
    
    Parameters:
    - X_train: Training data to fit the scaler
    - X_val: Validation data to be transformed based on X_train's scaling
    - X_testing: Testing data to be transformed based on X_train's scaling
    
    Returns:
    - Scaled X_train, X_val, and X_testing
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler on X_train and transform X_train
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Use the same scaler to transform X_val and X_testing
    X_val_scaled = scaler.transform(X_val)
    X_testing_scaled = scaler.transform(X_testing)
    
    return X_train_scaled, X_val_scaled, X_testing_scaled

def standardscaler(X_train, X_val=None, X_test=None):
    """
    Scales the input training, validation (if present), and test data using StandardScaler.
    """
    scaler = StandardScaler()

    # Fit scaler on training data
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data

    # Scale validation data if provided
    if X_val is not None and len(X_val) > 0:
        X_val_scaled = scaler.transform(X_val)  # Only transform validation data
    else:
        X_val_scaled = None  # No validation data to scale

    # Scale test data if provided
    if X_test is not None and len(X_test) > 0:
        X_testing_scaled = scaler.transform(X_test)  # Only transform test data
    else:
        X_testing_scaled = None  # No test data to scale

    return X_train_scaled, X_val_scaled, X_testing_scaled


def impute_nan_values(database):
    """
    Impute NaN values in the RSSI data of the database using the mean strategy.

    Args:
        database (dict): A dictionary containing 'trnrss' and 'tstrss' with NaN values.

    Returns:
        dict: Updated database with imputed RSSI values.
    """
    #  KNN Imputation
    knn_imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors

    # Apply KNN imputation to training and test RSSI data
    database['trnrss'] = knn_imputer.fit_transform(database['trnrss'])
    database['tstrss'] = knn_imputer.transform(database['tstrss'])
    

    # # Create an imputer object with the mean strategy
    # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # # Impute training RSSI data
    # trnrss_before = database['trnrss'].copy()  # Keep a copy for debugging
    # database['trnrss'] = imputer.fit_transform(database['trnrss'])

    # # Impute test RSSI data
    # tstrss_before = database['tstrss'].copy()  # Keep a copy for debugging
    # database['tstrss'] = imputer.transform(database['tstrss'])

    # # Check if any columns were entirely NaN
    # dropped_trnrss = np.setdiff1d(np.arange(trnrss_before.shape[1]), np.arange(database['trnrss'].shape[1]))
    # dropped_tstrss = np.setdiff1d(np.arange(tstrss_before.shape[1]), np.arange(database['tstrss'].shape[1]))

    # if len(dropped_trnrss) > 0:
    #     print(f"Warning: The following columns were dropped from 'trnrss': {dropped_trnrss.tolist()}")
    # if len(dropped_tstrss) > 0:
    #     print(f"Warning: The following columns were dropped from 'tstrss': {dropped_tstrss.tolist()}")

    return database

def custom_impute(database, default_value):
    database['trnrss'] = np.where(np.isnan(database['trnrss']), default_value, database['trnrss'])
    database['tstrss'] = np.where(np.isnan(database['tstrss']), default_value, database['tstrss'])
    return database


def calculate_3d_positioning_error(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))


##################################Below this, i saved the functions but still not using them


# Handling data representation: convert data to positive RSSI values
def data_rep_positive(database):
    min_rssi = min(database['trnrss'].min(), database['tstrss'].min())
    database['trnrss'] -= min_rssi
    database['tstrss'] -= min_rssi
    
    # Copy the labels (assuming they exist in the same structure)
    database['trncrd'] = database.get('trncrd', None)
    database['tstcrd'] = database.get('tstcrd', None)
    
    return database

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


import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, train_test_split, cross_val_score

from preprocessing import minmaxscaler


dataset_results = {}
def runSVR(processed_data, base_name, results_directory,dataset_params, config):#  C=1.0, kernel='rbf', epsilon=0.001, gamma=0.01, max_iter=300000):
    """Runs an SVR experiment on the processed data."""
    
    C = config.get('C', 1.0)
    kernel = config.get('kernel', 'rbf')
    epsilon = config.get('epsilon', 0.001)
    gamma = config.get('gamma', 0.01)
    train_test_splits = config.get('train_test_splits', 0.02)
    repetitions = config.get('repetitions', 10)
    max_iter = config.get('max_iter', 5000000)

    # Fetching non-detected values from dataset_params
    minValueDetected = dataset_params.get('minValueDetected', 0)
    defaultNonDetectedValue = dataset_params.get('defaultNonDetectedValue', 100)
    newNonDetectedValue = dataset_params.get('newNonDetectedValue', 0)
    
    rsamples1 = dataset_params.get('rsamples1', None)
    osamples1 = dataset_params.get('osamples1', None)
    nmacs1 = dataset_params.get('nmacs1', None)
    rsamples = dataset_params.get('rsamples', None)
    osamples = dataset_params.get('osamples', None)
    nmacs = dataset_params.get('nmacs', None)

    # Extract the training and testing data
    train_rssi = pd.DataFrame(processed_data['trnrss'])
    test_rssi = pd.DataFrame(processed_data['tstrss'])
    train_coords = pd.DataFrame(processed_data['trncrd'], columns=['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID'])
    test_coords = pd.DataFrame(processed_data['tstcrd'], columns=['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID'])

    # Concatenate the DataFrames
    train_df_combined = pd.concat([train_coords[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']], train_rssi], axis=1)
    X = train_df_combined.iloc[:, 5:] 
    y = train_df_combined[['Latitude', 'Longitude', 'Altitude']].values 

    test_df_combined = pd.concat([test_coords[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']], test_rssi], axis=1)
    X_testing = test_df_combined.iloc[:, 5:] 
    y_testing = test_df_combined[['Latitude', 'Longitude', 'Altitude']].values
    
    train_test_splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    repetitions = 10
    #best_config = {}
    best_mean_error_testing = float('inf')  # Initialize to infinity to find minimum
    best_val_error = float('inf')

    all_results = []
    
    for test_split in train_test_splits:
        #repetition_results = []   
        mean_errors_val_list = []
        mean_errors_testing_list = []
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, random_state=420)   
        X_train_scaled, X_val_scaled, X_testing_scaled = minmaxscaler(X_train, X_val, X_testing)

        for rep in range(repetitions):    
       
            # Train individual SVR models for Latitude, Longitude, Altitude
            svr_lat = SVR(C=C, kernel=kernel, epsilon=epsilon, gamma=gamma, max_iter=max_iter)
            svr_lon = SVR(C=C, kernel=kernel, epsilon=epsilon, gamma=gamma, max_iter=max_iter)
            svr_alt = SVR(C=C, kernel=kernel, epsilon=epsilon, gamma=gamma, max_iter=max_iter)
            
            # Fit each SVR model
            svr_lat.fit(X_train_scaled, y_train[:, 0])  # Latitude
            svr_lon.fit(X_train_scaled, y_train[:, 1])  # Longitude
            svr_alt.fit(X_train_scaled, y_train[:, 2])  # Altitude
            
            # Predict for validation
            y_pred_val_lat = svr_lat.predict(X_val_scaled)
            y_pred_val_lon = svr_lon.predict(X_val_scaled)
            y_pred_val_alt = svr_alt.predict(X_val_scaled)
            
            # Combine predictions for validation
            y_pred_val = np.vstack((y_pred_val_lat, y_pred_val_lon, y_pred_val_alt)).T
            errors_val = np.linalg.norm(y_val - y_pred_val, axis=1)
            mean_3d_error_val = np.round(np.mean(errors_val), 5)

            # Predict for testing
            y_pred_test_lat = svr_lat.predict(X_testing_scaled)
            y_pred_test_lon = svr_lon.predict(X_testing_scaled)
            y_pred_test_alt = svr_alt.predict(X_testing_scaled)

            # Combine predictions for testing
            y_pred_testing = np.vstack((y_pred_test_lat, y_pred_test_lon, y_pred_test_alt)).T
            errors_testing = np.linalg.norm(y_testing - y_pred_testing, axis=1)
            mean_3d_error_testing = np.round(np.mean(errors_testing), 5)

            mean_errors_testing_list.append(mean_3d_error_testing)
            mean_errors_val_list.append(mean_3d_error_val)

            print(f"Mean 3D Error Validation: Repetition {rep+1}/{repetitions}, Test size: {test_split}, validation error: {mean_3d_error_val:.5f} m, Testing error: {mean_3d_error_testing:.5f} m")
            
            # Store the average results for the current test size
            mean_error_val = np.mean(mean_errors_val_list)
            mean_error_testing = np.mean(mean_errors_testing_list)
   
            # Save individual errors to a CSV file
            errors_df = pd.DataFrame({
                'Sample_Index': range(len(errors_testing)),  # Optionally add sample indices
                'Repetition': rep + 1,
                '3D_Positioning_Error': np.round(errors_testing,5)
            })
   
            split_folder = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err','svm_plain2024', base_name,
                                        f"kernel={kernel}",
                                         f"C={C}", 
                                        f"gamma={gamma}",
                                        f"split_{test_split}")
            
            if not os.path.exists(split_folder):
                os.makedirs(split_folder)

            # Save the individual errors to CSV
            individual_errors_file = os.path.join(split_folder , f"error_repetition{rep+1}.csv")#, f"errors_{base_name}_C{best_params['C']}.csv")#
            errors_df.to_csv(individual_errors_file, index=False)

            # Save predictions (longitude, latitude, altitude) to a CSV file
            predictions_df = pd.DataFrame(y_pred_testing, columns=['Longitude', 'Latitude', 'Altitude'])
            predictions_file = os.path.join(split_folder , f"predictions_repetition{rep+1}.csv")#, f"predictions_{base_name}_C{best_params['C']}.csv")#
            predictions_df.to_csv(predictions_file, index=False)
            
            # Calculate average errors over all iterations for this dataset
            mean_error_val = np.mean(mean_error_val) if mean_error_val else float('nan')
            mean_error_testing = np.mean(mean_error_testing) if mean_error_testing else float('nan')

            # Check if this is the best configuration so far
            if mean_3d_error_testing < best_mean_error_testing:
                best_mean_error_testing = mean_3d_error_testing
            if mean_3d_error_val < best_val_error:
                best_val_error = mean_3d_error_val
            
                dataset_results[base_name] = {
                    'Dataset': base_name,
                    'Test_Split': test_split,
                    # 'C': C,
                    # 'Kernel': kernel,
                    # 'Gamma': gamma,
                    'Error_val': best_val_error,
                    'Testing_Error': best_mean_error_testing,                 
                }

    # Save combined results to a single CSV file
    combined_results_df = pd.DataFrame(dataset_results.values())
    combined_csv_path = os.path.join(results_directory, 'combined_mean_errors.csv')
    combined_results_df.to_csv(combined_csv_path, index=False)
    #print(f"Saved combined mean errors summary to {combined_csv_path}")
    
    print(f"\nRunning the algorithm on: {base_name}")#{result['dataset']}")
    print(f'    database features pre  : [{rsamples1},{osamples1},{nmacs1}]')
    print(f'    database New features  : [{rsamples},{osamples},{nmacs}]')
    print(f'    C                      : {C}')
    print(f'    kernel                 : {kernel}')
    print(f'    gamma                  : {gamma}')
    print(f'    minValueDetected       : {minValueDetected }')
    print(f'    defaultNonDetectedValue: {defaultNonDetectedValue}')
    print(f'    newNonDetectedValue    : {newNonDetectedValue}')
    print(f"Avg 3D Positioning Error Validation  : {best_val_error:.5f} m")
    print(f"Avg 3D Positioning Error Testing: {best_mean_error_testing:.5f} m\n")


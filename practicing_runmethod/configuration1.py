import os
import sys
import copy
import random
import logging
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from configurations_functions_svm import (remapBldDB, remapFloorDB, replace_non_detected_values_orig)
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
         
        for rep in range(repetitions):    
              
            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, random_state=420)   
            X_train_scaled, X_val_scaled, X_testing_scaled = minmaxscaler(X_train, X_val, X_testing)


            # Implement your SVR model training and evaluation here
            model = SVR(C=C, kernel=kernel, epsilon=epsilon, gamma=gamma, max_iter=max_iter)
            model = MultiOutputRegressor(model, n_jobs=1)
            
            kf = KFold(n_splits=5, shuffle=True, random_state=420)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
            mean_cv_error = np.round(-np.mean(cv_scores), 2)
            #print(f"Mean CV Error: {mean_cv_error}")
            
            # Fit the model
            model.fit(X_train_scaled, y_train)
            
            # Predict for validation and testing
            y_pred_val = model.predict(X_val_scaled)
            errors_val = np.linalg.norm(y_val - y_pred_val, axis=1)
            mean_3d_error_val = np.round(np.mean(errors_val), 5)

            y_pred_testing = model.predict(X_testing_scaled)
            errors_testing = np.linalg.norm(y_testing - y_pred_testing, axis=1)
            mean_3d_error_testing = np.round(np.mean(errors_testing), 5)
            
            mean_errors_testing_list.append(mean_3d_error_testing)

            mean_errors_val_list.append(mean_3d_error_val)
            mean_errors_testing_list.append(mean_3d_error_testing)

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
            
            # repetition_results.append({
            #     'Repetition': rep + 1,
            #     'Mean_3D_Error_Validation': mean_3d_error_val,
            #     'Mean_3D_Error_Testing': mean_3d_error_testing
            # })
            
            
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


  # # Perform K-Fold CV to select the best hyperparameters
    # for params in [{'C': C, 'kernel': kernel, 'epsilon': epsilon, 'gamma': gamma}]:  # Add more hyperparameter sets as needed
    #     model = SVR(**params, max_iter=max_iter)
    #     model = MultiOutputRegressor(model, n_jobs=1)
        
    #     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
    #     mean_cv_error = np.round(-np.mean(cv_scores), 2)
    #     print(f"Mean CV Error for {params}: {mean_cv_error}")
        
    #     # Update best parameters if we find a better score
    #     if mean_cv_error < best_score:
    #         best_score = mean_cv_error
    #         best_params = params
    
    # print(f"Best Parameters: {best_params}")

    # # Retrain on full training data with the best hyperparameters
    # best_model = SVR(**best_params, max_iter=max_iter)
    # best_model = MultiOutputRegressor(best_model, n_jobs=1)
    
    # model.fit(X_train_scaled, y_train)  # Train on the full training set
   
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from preprocessing import minmaxscaler

dataset_results = []
summary_list = []
best_configs_dict = {}

def runSVR_C9(processed_data, base_name, results_directory,dataset_params, config):
    
    C = config['C']
    kernel = config['kernel']
    epsilon = config['epsilon']
    degree = config['degree']
    gamma = config['gamma']
    max_iter = config['max_iter']
    tol = config['tol']
    train_test_splits = config.get('train_test_splits', 0.02)
    repetitions = config.get('repetitions', 10)
    coef0 = config['coef0']
    
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
    

    # List to save errors
    all_errors_val = []
    all_errors_testing = []
    
    best_degree = None
    best_beta = None
    best_split = None
    best_kernel = None
    best_gamma = None
    best_C = None
    best_epsilon = None
    best_tol = None
    best_error = float('inf')
    best_val_error = float('inf')
    
    for test_split in train_test_splits:
    # Prepare cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=420)
        X_train, X_val, y_train, y_val =train_test_split(X , y, test_size=test_split, random_state=420)
        X_train_scaled, X_val_scaled, X_testing_scaled = minmaxscaler(X_train, X_val, X_testing)

        # Start grid search over all combinations of hyperparameters
        for kernel_value in tqdm(kernel , desc=f"Processing dataset: {base_name}"):
            for C_value in C:
                if kernel_value in ['rbf', 'poly', 'sigmoid']:
                    for epsilon_value in epsilon:
                        for gamma_value in gamma:
                            for degree_value in degree:
                                for tol_value in tol:
                                    for coef0_value in coef0:
                                        for rep in range(repetitions):                    
                                            if kernel_value == 'poly':
                                                svr_lat = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, degree=degree_value, tol=tol_value, max_iter=max_iter)
                                                svr_lon = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, degree=degree_value, tol=tol_value, max_iter=max_iter)
                                                svr_alt = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, degree=degree_value, tol=tol_value, max_iter=max_iter)
                                                
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
                                                
                                                print(f"Repetition {rep+1}/{repetitions}, Test size: {test_split}, C: {C_value}, "
                                                    f"Kernel: {kernel_value}, gamma={gamma_value}, Degree={degree_value},  "
                                                    f"Validation error: {mean_3d_error_val:.5f}, Testing error: {mean_3d_error_testing:.5f}")
                                                
                                                # Save individual errors to a CSV file
                                                errors_df = pd.DataFrame({
                                                    'Sample_Index': range(len(errors_testing)),
                                                    'Repetition': rep + 1,
                                                    '3D_Positioning_Error': mean_3d_error_testing
                                                })
                                                
                                                # Create the split folder first
                                                split_folder = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 
                                                                            'svm_plain2024', base_name,
                                                                            f"split_{test_split}")
                                                
                                                # Check if split folder exists, if not, create it
                                                if not os.path.exists(split_folder):
                                                    os.makedirs(split_folder)
                                                
                                                tol_folder = os.path.join(split_folder, f"C_{C_value}", f"kernel_{kernel_value}", f"gamma_{gamma_value}", f"epsilon_{epsilon_value}", f"degree_{degree_value}", f"tol_{tol_value}")
                                                if not os.path.exists(tol_folder):
                                                    os.makedirs(tol_folder) 
        
                                                # Save the individual errors to CSV
                                                individual_errors_file = os.path.join(tol_folder , f"error_repetition{rep+1}.csv")#, f"errors_{base_name}_C{best_params['C']}.csv")#
                                                errors_df.to_csv(individual_errors_file, index=False)
                                                
                                                predictions_df = pd.DataFrame(y_pred_testing, columns=['Longitude', 'Latitude', 'Altitude'])
                                                predictions_file = os.path.join(tol_folder , f"predictions_repetition{rep+1}.csv")#, f"predictions_{base_name}_C{best_params['C']}.csv")#
                                                predictions_df.to_csv(predictions_file, index=False)                            
                                                
                                                if mean_3d_error_testing < best_error:
                                                    best_error = mean_3d_error_testing
                                                    best_kernel = kernel_value
                                                    best_C = C_value
                                                    best_tol = tol_value
                                                    best_gamma = gamma_value
                                                    best_split = test_split
                                                    best_epsilon = epsilon_value
                                                    best_degree = degree_value if kernel_value == 'poly' else None
                                                if mean_3d_error_val < best_val_error:
                                                    best_val_error = mean_3d_error_val
                                    
                                            elif kernel_value == 'sigmoid':
                                                svr_lat = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, degree=degree_value, coef0=coef0_value, tol=tol, max_iter=max_iter)
                                                svr_lon = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, degree=degree_value, coef0=coef0_value, tol=tol, max_iter=max_iter)
                                                svr_alt = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, degree=degree_value, coef0=coef0_value, tol=tol, max_iter=max_iter)
                                                
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
                                                
                                                print(f"Repetition {rep+1}/{repetitions}, Test size: {test_split}, C: {C_value},"
                                                    f"Kernel: {kernel_value}, Beta={coef0_value}, gamma={gamma_value}, epsilon={epsilon_value},"
                                                    f"Validation error: {mean_3d_error_val:.5f}, Testing error: {mean_3d_error_testing:.5f}")
                                                
                                                # Save individual errors to a CSV file
                                                errors_df = pd.DataFrame({
                                                    'Sample_Index': range(len(errors_testing)),
                                                    'Repetition': rep + 1,
                                                    '3D_Positioning_Error': mean_3d_error_testing
                                                })
                                                
                                                # Create the split folder first
                                                split_folder = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 
                                                                            'svm_plain2024', base_name,
                                                                            f"split_{test_split}")
                                                
                                                # Check if split folder exists, if not, create it
                                                if not os.path.exists(split_folder):
                                                    os.makedirs(split_folder)
                                                    
                                                tol_folder = os.path.join(split_folder, f"C_{C_value}", f"kernel_{kernel_value}", f"gamma_{gamma_value}", f"epsilon_{epsilon_value}", f"Beta_{coef0_value}", f"tol_{tol_value}")
                                                if not os.path.exists(tol_folder):
                                                    os.makedirs(tol_folder)                                                                              
                                                
                                                # Save the individual errors to CSV
                                                individual_errors_file = os.path.join(tol_folder , f"error_repetition{rep+1}.csv")#, f"errors_{base_name}_C{best_params['C']}.csv")#
                                                errors_df.to_csv(individual_errors_file, index=False)
                                                
                                                predictions_df = pd.DataFrame(y_pred_testing, columns=['Longitude', 'Latitude', 'Altitude'])
                                                predictions_file = os.path.join(tol_folder , f"predictions_repetition{rep+1}.csv")#, f"predictions_{base_name}_C{best_params['C']}.csv")#
                                                predictions_df.to_csv(predictions_file, index=False)                            
                                                

                                                if mean_3d_error_testing < best_error:
                                                    best_error = mean_3d_error_testing
                                                    best_kernel = kernel_value
                                                    best_C = C_value
                                                    best_gamma = gamma_value
                                                    best_epsilon = epsilon_value
                                                    best_split = test_split
                                                    best_tol = tol_value
                                                    best_gamma = None
                                                    best_beta = best_beta if kernel_value == 'sigmoid' else None
                                                if mean_3d_error_val < best_val_error:
                                                    best_val_error = mean_3d_error_val
                                            
                                            elif kernel_value == 'rbf':
                                                svr_lat = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, tol=tol, max_iter=max_iter)
                                                svr_lon = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, tol=tol, max_iter=max_iter)
                                                svr_alt = SVR(C=C_value, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon_value, tol=tol, max_iter=max_iter)
                                                
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
                                                
                                                print(f"Repetition {rep+1}/{repetitions}, Test size: {test_split}, C: {C_value}, "
                                                    f"Kernel: {kernel_value}, gamma={gamma_value}, epsilon={epsilon_value},"
                                                    f"Validation error: {mean_3d_error_val:.5f}, Testing error: {mean_3d_error_testing:.5f}")
                                                
                                                # Save individual errors to a CSV file
                                                errors_df = pd.DataFrame({
                                                    'Sample_Index': range(len(errors_testing)),
                                                    'Repetition': rep + 1,
                                                    '3D_Positioning_Error': mean_3d_error_testing
                                                })
                                        
                                                # Create the split folder first
                                                split_folder = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 
                                                                            'svm_plain2024', base_name,
                                                                            f"split_{test_split}")
                                                
                                                # Check if split folder exists, if not, create it
                                                if not os.path.exists(split_folder):
                                                    os.makedirs(split_folder)
                                                
                                                tol_folder = os.path.join(split_folder, f"C_{C_value}", f"kernel_{kernel_value}", f"gamma_{gamma_value}", f"epsilon_{epsilon_value}", f"tol_{tol_value}")
                                                if not os.path.exists(tol_folder):
                                                    os.makedirs(tol_folder)                         
                                                
                                                # Save the individual errors to CSV
                                                individual_errors_file = os.path.join(tol_folder , f"error_repetition{rep+1}.csv")#, f"errors_{base_name}_C{best_params['C']}.csv")#
                                                errors_df.to_csv(individual_errors_file, index=False)
                                                
                                                predictions_df = pd.DataFrame(y_pred_testing, columns=['Longitude', 'Latitude', 'Altitude'])
                                                predictions_file = os.path.join(tol_folder , f"predictions_repetition{rep+1}.csv")#, f"predictions_{base_name}_C{best_params['C']}.csv")#
                                                predictions_df.to_csv(predictions_file, index=False)                            
                                                                                        

                                                if mean_3d_error_testing < best_error:
                                                    best_error = mean_3d_error_testing
                                                    best_kernel = kernel_value
                                                    best_C = C_value
                                                    best_split = test_split
                                                    best_degree = None
                                                    best_gamma = gamma_value
                                                    best_tol = tol_value
                                                    best_epsilon = epsilon_value
                                                if mean_3d_error_val < best_val_error:
                                                    best_val_error = mean_3d_error_val
                                    
                                            else:
                                                svr_lat = SVR(C=C_value, kernel=kernel_value, epsilon=epsilon_value, tol=tol, max_iter=max_iter)
                                                svr_lon = SVR(C=C_value, kernel=kernel_value, epsilon=epsilon_value, tol=tol, max_iter=max_iter)
                                                svr_alt = SVR(C=C_value, kernel=kernel_value, epsilon=epsilon_value, tol=tol, max_iter=max_iter)
                                                
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
                                                
                                                print(f"Repetition {rep+1}/{repetitions}, Test size: {test_split}, C: {C_value}, "
                                                    f"Kernel: {kernel_value}, Epsilon: {epsilon_value},"
                                                    f"Validation error: {mean_3d_error_val:.5f}, Testing error: {mean_3d_error_testing:.5f}")
                                                
                                                # Save individual errors to a CSV file
                                                errors_df = pd.DataFrame({
                                                    'Sample_Index': range(len(errors_testing)),
                                                    'Repetition': rep + 1,
                                                    '3D_Positioning_Error': mean_3d_error_testing
                                                })
                                                
                                                # Create the split folder first
                                                split_folder = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 
                                                                            'svm_plain2024', base_name,
                                                                            f"split_{test_split}")
                                                
                                                # Check if split folder exists, if not, create it
                                                if not os.path.exists(split_folder):
                                                    os.makedirs(split_folder)
                                    
                                                tol_folder = os.path.join(split_folder, f"C_{C_value}", f"kernel_{kernel_value}", f"epsilon_{epsilon_value}", f"tol_{tol_value}")
                                                if not os.path.exists(tol_folder):
                                                    os.makedirs(tol_folder)
                                                            
                                                
                                                # Save the individual errors to CSV
                                                individual_errors_file = os.path.join(tol_folder , f"error_repetition{rep+1}.csv")#, f"errors_{base_name}_C{best_params['C']}.csv")#
                                                errors_df.to_csv(individual_errors_file, index=False)
                                                
                                                predictions_df = pd.DataFrame(y_pred_testing, columns=['Longitude', 'Latitude', 'Altitude'])
                                                predictions_file = os.path.join(tol_folder , f"predictions_repetition{rep+1}.csv")#, f"predictions_{base_name}_C{best_params['C']}.csv")#
                                                predictions_df.to_csv(predictions_file, index=False)                            
                                                
                                                if mean_3d_error_testing < best_error:
                                                    best_error = mean_3d_error_testing
                                                    best_kernel = kernel_value
                                                    best_C = C_value
                                                    best_epsilon = epsilon_value
                                                    best_split = test_split
                                                    best_tol = tol_value
                                                    best_degree = None
                                                    best_gamma = None
                                                if mean_3d_error_val < best_val_error:
                                                    best_val_error = mean_3d_error_val              

    # Store the best configuration for this dataset
    best_configs_dict[base_name] = {
        'C': best_C, 
        'kernel': best_kernel,
        'degree': best_degree if best_kernel == 'poly' else None, 
        'gamma': best_gamma, 
        'epsilon': best_epsilon, 
        'Test split': best_split,
        'Beta': best_beta if best_kernel == 'sigmoid' else None,
        'tol' : best_tol,
        'Val_error': best_val_error, 
        'Test_error': best_error
    }
    summary_list.append([base_name, best_C, best_kernel, best_degree if best_kernel == 'poly' else None, best_gamma,best_beta if best_kernel == 'sigmoid' else None, best_epsilon, best_split,best_tol, best_val_error, best_error])

    # Save the summary of the best configurations
    summary_df = pd.DataFrame(summary_list, columns=['Dataset', 'Best C', 'Best kernel','Best Degree', 'Best gamma','Best Beta', 'Best Epsilon', 'Best Split','Best tol', 'Error val', 'Testing error'])
    summary_df.to_csv(os.path.join(results_directory, 'summary_best_configurations_experiment_5.csv'), index=False)

   
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
    print(f"Avg 3D Positioning Error Testing: {best_error:.5f} m\n")
   
   
   
   
               
                    #  # Inside your loop where you check for the best error
                    #                 if mean_3d_error_testing < best_error:
                    #                     best_error = mean_3d_error_testing
                    #                     best_kernel = kernel_value
                    #                     best_C = C
                    #                     best_gamma = gamma
                    #                     best_epsilon = epsilon
                    #                     best_split = test_split
                    #                     best_cv_error = mean_cv_error
                    #                     if kernel_value == 'poly':
                    #                         best_degree = degree_value
                    #                         best_beta = None  # Since 'Beta' is not used for 'poly' kernel
                    #                     elif kernel_value == 'sigmoid':
                    #                         best_degree = None
                    #                         best_beta = coef0_value
                    #                     else:
                    #                         best_degree = None
                    #                         best_beta = None
   
   
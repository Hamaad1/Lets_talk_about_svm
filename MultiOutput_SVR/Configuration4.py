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


def runSVR_C4(processed_data, base_name, results_directory,dataset_params, config):
    
    C = config.get('C', 1.0)
    kernel = config['kernel']
    gamma = config.get('gamma','scale')
    epsilon = config.get('epsilon', 0.001)
    tol = config.get('tol', 0.001)
    degree = config.get('degree', 3)
    coef0 = config.get('coef0', 0.0)
    train_test_splits = config.get('train_test_splits', 0.02)
    repetitions = config.get('repetitions', 10)
    max_iter = config.get('max_iter', 1000) 
    
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
      
    best_configs_dict = {} #remove these 3 line to revert
   
    best_kernel = None
    best_error = float('inf')
    best_val_error = float('inf')

    # Best configuration tracking variables
    best_degree = None
    best_beta = None
    best_split = None
    best_gamma = None
    best_epsilon = None
    best_tol = None
     
    # Grid search over parameters
    for test_split in train_test_splits:
        # Prepare cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=420)
        X_train, X_val, y_train, y_val =train_test_split(X , y, test_size=test_split, random_state=420)
        X_train_scaled, X_val_scaled, X_testing_scaled = minmaxscaler(X_train, X_val, X_testing)
    
        for kernel_value in tqdm(kernel, desc=f"Processing dataset: {base_name} with varying kernel and gamma"):
            if kernel_value in ['rbf', 'poly', 'sigmoid', 'linear']:
                for gamma_value in gamma:
                    for rep in range(repetitions):
                        
                        if kernel_value == 'poly':                   
                            model = SVR(C=C, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon, degree=degree, coef0=coef0, tol=tol, max_iter=max_iter)
                            model = MultiOutputRegressor(model, n_jobs=-1)

                            # Perform cross-validation
                            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
                            mean_cv_error = np.round(-np.mean(cv_scores), 2)
                            model.fit(X_train_scaled, y_train)

                            y_pred_val = model.predict(X_val_scaled)
                            errors_val = np.linalg.norm(y_val - y_pred_val, axis=1)
                            mean_3d_error_val = np.round(np.mean(errors_val), 5)

                            y_pred_testing = model.predict(X_testing_scaled)
                            errors_testing = np.linalg.norm(y_testing - y_pred_testing, axis=1)
                            mean_3d_error_testing = np.round(np.mean(errors_testing), 5)
                            print(f"Processing: Test size: {test_split}, Repetition {rep+1}/{repetitions}, kernel:{kernel_value}, gamma:{gamma_value}, degree:{degree}, beta:{coef0}, validation error: {mean_3d_error_val:.5f} m, Testing error: {mean_3d_error_testing:.5f} m")
                            
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
                            
                            # Now create the kernel folder inside the split folder
                            gamma_folder = os.path.join(split_folder, f"C_{C}", f"kernel_{kernel_value}", f"gamma_{gamma_value}")
                            if not os.path.exists(gamma_folder):
                                os.makedirs(gamma_folder)
                            
                            # Save the individual errors to CSV
                            individual_errors_file = os.path.join(gamma_folder , f"error_repetition{rep+1}.csv")#, f"errors_{base_name}_C{best_params['C']}.csv")#
                            errors_df.to_csv(individual_errors_file, index=False)
                            
                            predictions_df = pd.DataFrame(y_pred_testing, columns=['Longitude', 'Latitude', 'Altitude'])
                            predictions_file = os.path.join(gamma_folder , f"predictions_repetition{rep+1}.csv")#, f"predictions_{base_name}_C{best_params['C']}.csv")#
                            predictions_df.to_csv(predictions_file, index=False)                            
                            
                            # Update the best epsilon and error if current test error is lower
                            if mean_3d_error_testing < best_error:  
                                best_error = mean_3d_error_testing
                                best_kernel = kernel_value
                                best_gamma = gamma_value
                                best_epsilon = epsilon
                                best_split = test_split
                                best_degree = degree if kernel_value == 'poly' else None
                                best_beta = coef0 if kernel_value == 'poly' else None 
                            if mean_3d_error_val < best_val_error:
                                best_val_error = mean_3d_error_val  # Update to the new best validation error if it's better 
                        
                        elif kernel_value == 'sigmoid':
                            model = SVR(C=C, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon, coef0=coef0, tol=tol, max_iter=max_iter)
                            model = MultiOutputRegressor(model, n_jobs=-1)

                            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
                            mean_cv_error = np.round(-np.mean(cv_scores), 2)
                            model.fit(X_train_scaled, y_train)

                            y_pred_val = model.predict(X_val_scaled)
                            errors_val = np.linalg.norm(y_val - y_pred_val, axis=1)
                            mean_3d_error_val = np.round(np.mean(errors_val), 5)

                            y_pred_testing = model.predict(X_testing_scaled)
                            errors_testing = np.linalg.norm(y_testing - y_pred_testing, axis=1)
                            mean_3d_error_testing = np.round(np.mean(errors_testing), 5)
                            print(f"Processing: Test size: {test_split}, Repetition {rep+1}/{repetitions}, kernel:{kernel_value}, gamma:{gamma_value}, validation error: {mean_3d_error_val:.5f} m, Testing error: {mean_3d_error_testing:.5f} m")
                                                
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
                            
                            # Now create the kernel folder inside the split folder
                            kernel_folder = os.path.join(split_folder, f"kernel_{kernel_value}")
                            if not os.path.exists(kernel_folder):
                                os.makedirs(kernel_folder)
                            
                            # Save the individual errors to CSV
                            individual_errors_file = os.path.join(kernel_folder , f"error_repetition{rep+1}.csv")#, f"errors_{base_name}_C{best_params['C']}.csv")#
                            errors_df.to_csv(individual_errors_file, index=False)
                            
                            predictions_df = pd.DataFrame(y_pred_testing, columns=['Longitude', 'Latitude', 'Altitude'])
                            predictions_file = os.path.join(kernel_folder , f"predictions_repetition{rep+1}.csv")#, f"predictions_{base_name}_C{best_params['C']}.csv")#
                            predictions_df.to_csv(predictions_file, index=False)
                            
                            #if kernel_value in ['poly', 'sigmoid']: #remove this to revert
                            if mean_3d_error_testing < best_error:
                                best_error = mean_3d_error_testing
                                best_kernel = kernel_value
                                best_split = test_split
                                best_gamma = gamma
                                best_epsilon = epsilon
                                best_beta = coef0 if kernel_value == 'poly' else None
                                # Update validation error if it's better
                            if mean_3d_error_val < best_val_error:
                                best_val_error = mean_3d_error_val  # Update to the new best validation error if it's better
            
                        else:
                            model = SVR(C=C, kernel=kernel_value, gamma=gamma_value, epsilon=epsilon, tol=tol, max_iter=max_iter)
                            model = MultiOutputRegressor(model, n_jobs=-1)
                        
                            # Perform cross-validation and fit
                            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
                            mean_cv_error = np.round(-np.mean(cv_scores), 2)
                            model.fit(X_train_scaled, y_train)
                            
                            y_pred_val = model.predict(X_val_scaled)
                            errors_val = np.linalg.norm(y_val - y_pred_val, axis=1)
                            mean_3d_error_val = np.round(np.mean(errors_val), 5)

                            y_pred_testing = model.predict(X_testing_scaled)
                            errors_testing = np.linalg.norm(y_testing - y_pred_testing, axis=1)
                            mean_3d_error_testing = np.round(np.mean(errors_testing), 5)
                            print(f"Processing: Test size: {test_split}, Repetition {rep+1}/{repetitions}, kernel:{kernel_value}, Gamma:{gamma_value}, validation error: {mean_3d_error_val:.5f} m, Testing error: {mean_3d_error_testing:.5f} m")
                            
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
                            
                            # Now create the kernel folder inside the split folder
                            gamma_folder = os.path.join(split_folder, f"C_{C}", f"kernel_{kernel_value}", f"gamma_{gamma_value}")
                            if not os.path.exists(gamma_folder):
                                os.makedirs(gamma_folder)
                            
                            # Save the individual errors to CSV
                            individual_errors_file = os.path.join(gamma_folder , f"error_repetition{rep+1}.csv")#, f"errors_{base_name}_C{best_params['C']}.csv")#
                            errors_df.to_csv(individual_errors_file, index=False)
                            
                            predictions_df = pd.DataFrame(y_pred_testing, columns=['Longitude', 'Latitude', 'Altitude'])
                            predictions_file = os.path.join(gamma_folder , f"predictions_repetition{rep+1}.csv")#, f"predictions_{base_name}_C{best_params['C']}.csv")#
                            predictions_df.to_csv(predictions_file, index=False)                            

                            if mean_3d_error_testing < best_error:
                                best_error = mean_3d_error_testing
                                best_kernel = kernel_value
                                best_degree = None 
                                best_beta = None  
                                best_gamma = gamma_value
                                best_split = test_split
                                best_epsilon = epsilon
                            if mean_3d_error_val < best_val_error:
                                best_val_error = mean_3d_error_val  
        
    # Store best configuration
    best_configs_dict[base_name] = {
        'C': C, 
        'kernel': best_kernel, 
        'degree': best_degree if best_kernel == 'poly' else None,
        'gamma': best_gamma, 
        'epsilon': best_epsilon, 
        'Beta': best_beta if best_kernel == 'sigmoid' else None,
        'Test Split': best_split,
        'Val_error': best_val_error, 
        'Test_error': best_error,
        'Tolerance': best_tol
        }
    summary_list.append([base_name, C, best_kernel, best_degree if best_kernel == 'poly' else None, best_gamma, best_beta if best_kernel == 'sigmoid' else None, best_epsilon, best_split,best_tol, best_val_error, best_error])

    # Save the summary
    summary_df = pd.DataFrame(summary_list, columns=['Dataset', 'C', 'Best kernel', 'Best Degree', 'Best gamma','Best Beta', 'Best Epsilon','Best Split','Best Tol', 'Error Val', 'Testing error'])
    summary_df.to_csv(os.path.join(results_directory, 'summary_best_configurations_experiment_3.csv'), index=False)

    #print("Completed grid search and saved results.")
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

   
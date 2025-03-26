
#Directories will be generated if not present during the exaustive search and then during the analysis.
root_dir : /mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/dataset
Input Directory: /mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Results_simulations
Output Directory: /mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Results and Analysis

    ###################################################################################################################################################
    A.  Simulation files to get total results i.e. plainSVRm_2024.py and plainSVRs_2024.py 
    ###################################################################################################################################################
    Path: /mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/
    
    Purpose: These files are used to run the final big configuration i.e., C8 to perform exaustive search for SVR for both Multioutput_SVR and Singleoutput_SVR

    How to execute:
    There are two method to execute these files.
        1. Run the files directly from the python environment as they also behave as an independent file to run execution (both files : plainSVRm_2024.py and plainSVRs_2024.py)
        2. You can use runmethod_cmd.py file to execute these files as well. it is also present in the same directory.

        Commands for the first run is also added in the end of each file as a template for execution.
    
    These files will read the data from the dataset folder from teh root directory i.e., root_dir, do the execution. It will perform multiple tasks in it requered for simulation. 
    It will keep the record of each combination in progress_file.json so that if the system crashes, it will skip the already executed files. Even if any combination form 
    the json file or the results form the input directory was removed mistakenly, then it will just execute tehat relevent combination from the previous combinations.


    ###################################################################################################################################################
    B.  Lets Talk about the Functions used in getCi_cmd.py: A command based analysis in analysis folder
    ###################################################################################################################################################
    
    File Path: /mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/Analysis/getCi_cmd.py
    Purpose: Analysis based on the given results form Input directory and the results will be saved in output directory.
    Command :  python getCi_cmd.py --config_name C1 C2 or any --datasets DSI2 MAN2 --subfolders plainSVRs_2024 plainSVRm_2024

    ###################################################################################################################################################
    Note:
    These are all the files used for the analysis purpose, it will take the data from the Results_simulations folder, generated using plainSVRm_2024.py and plainSVRs_2024.py 
    form Lets_talk_about_svm folder. These fiels can be ececuted independently or using runmethod.py from Lets_talk_about_svm folder using commands.


    It is the file for command based results and analysis and it will generate output in output directory for each configuration C1 to C8 is a structure like:
    output directory/plainSVRs_2024/config_name/dataset_name/configuration_folder/error_rep*.csv and prediction_rep*.csv file

    Command description used for analysis:
            It will accept config names like C1,C2,C3, to C8 , dataset name like DSI1,DSI2 (more then one names), 
            and subfolder and generated during results and simulation i.e. (plainSVRs_2024, plainSVRm_2024)

    Functions used in getCi_cmd.py:
    1. copy_config_files()
        This function is reading the relevant combination files of the given input configuration from input directory and then it is saving the relevent files in the output directory.
        It will generate a folder structure like : output directory/plainSVRs_2024/config_name/dataset_name/configuration_folder/error_rep*.csv and prediction_rep*.csv file.
        This function is commented out in save_summary_files(), as it take a little time to copy for big configuration, uncomment it first time to get the files,and then comment it again to save time.

    2. save_summary_files()
        This function is generating two summary files for each given input like mentione in command (plainSVRs_2024 plainSVRm_2024)
        1. First will save outputs i.e., Dataset name, mean error and standard deviation for given configurations like: Dataset,Mean Error,Standard Deviation
        2. It will save the values of the best configuration of the given data like:  Dataset,C,Kernel,Gamma,Epsilon,Tolerance,Mean Error,Standard Deviation

    3. process_error_files()
        This function is reading all the error files and processing them to calculate the mean error and standard deviation.

    4. find_best_configurations()
        This is the main function that is doing all the analysis and process
    

    ###################################################################################################################################################
    C.  Static file to execute directly= getC1.py to getC8.py
    ###################################################################################################################################################

    If you dont want to use command line than use getC1.py to getC8.py and or replace the value of config_name = "C1 to C8" in getC1.py, with required configuration name, it will do the same. 
    keep in mind , also adjust repetition number if using different repetitions.

    It can be posible that , you can generate 8 files for each configuration. so that you dont have to change the value of config_name = to any config name eg "C3"

    If you want to save the files as well from the final configuration then uncomment the save_repetition_files() from save_summary_files() fuction, It is just needed once. 
    Then comment it to save time.

    ###################################################################################################################################################
    D.  Command based file to execute independet combinations= getc1_cmd.py:  renamed as getCi_cmd.py
    ###################################################################################################################################################

    1. getc1_cmd.py

    This file is for command base execution to check the output of any individual combinations.
    To execute this, change the directory of the folder to make the command short as "python getc1_cmd.py 0.2 DSI1 1.0 rbf scale 0.001 3 0.001 0 5"   use without "". else use the long path as mentiond below
    
    How to change directory:  
    cd "/mnt/c/Users/hamaa/OneDrive - Università degli Studi di Catania/PHD code/transformer_practice/University_valencia_IP/Lets_talk_about_svm/MultiOutput_SVR/Analysis/getc1_cmd.py"
    
    Then type: python getc1_cmd.py 0.2 DSI1 1.0 rbf scale 0.001 3 0.001 0 5
    
    Numbers are the parameter in a sequece used in runmethod_cmd() function
    EG: Base Name: DSI1,  Split: 0.2, C: 1.0, Kernel: rbf, Gamma: scale, Epsilon: 0.001, Tol: 0.001, Repetitions: 5


    ###################################################################################################################################################
    E.   getCi_master_copy.py 
    ###################################################################################################################################################
    It is the maseter copy for the static files.
    In this file i am doing the analysis of the individually given combination, so that i can get the output analysis for the given configuration only



    ###################################################################################################################################################
    F.   Lets talk about helping files that contain the helping function
    ###################################################################################################################################################

    helping_functions_svm.py
        It has multiple functions required to execte the main files i.e. plainSVRm_2024.py and plainSVRs_2024.py
        Functions: fit_and_evaluate_model, prepare_data, save_results, save_progress, load_progress, files_exist, generate_param_combinations
    
    processing.py
        This file contains the processing function required to process the data.
        Functions: minmaxscaler, process_datasets

    optimization_function.py
        I am not using this file yet, However it contains some optimization functions.


    ###################################################################################################################################################
    G.   Lets talk about: getBestParams_cmd_individual.py
    ###################################################################################################################################################

    Command:python getBestParams_cmd_individual.py -fold plainSVRm_2024 -ts 0.2 -DS DSI2 -C 1.0 -K rbf -g scale -e 0.001 -d 3 -t 0.001 -coe 0.0 -rep 2


    ###################################################################################################################################################
    H.   Lets talk about json, log and pkl Files
    ###################################################################################################################################################
    progress_file.json
        It is the json file that contains the progress of the models. It records each processed combinations during exaustive search. It will help us in tracking the  combinations
    so that when the system crash or interupted etc then we can resuume from teh last recorded combination while skipping the previous combinations.
    
    config.json:
    It is the json file that contains the configuration of the system. I just recording the final configuration here. But i am not using it.

    svr_run_log
        This file is recording thelogs or the errors of the executions.

    progress_pkl
        It is same as progress_file.json, but i am not using it.


A: runmethod_cmd.py
This file is used for command based execution regarding the user based parameter
To use this file first change the directory as cd "right click the folder that has this file to copy path"
press enter.
Run this current configuration for configuration 1: python runmethod_cmd.py DSI1 1.0 rbf scale 0.001 0.001 0.2 1 2
execution header is: python run_experiments.py --base_name 'DSI1' --kernel 'rbf' --gamma 'scale' --epsilon 0.001 --tol 0.001 --train_test_split 0.2 --config_choice 1 --repetitions 2
It will take the initial defaults values by it self like, processed datast, redult_directory, data_directory, log_file path.

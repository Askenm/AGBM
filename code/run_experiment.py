# Miscellaneaous
import sys
import os
from datetime import datetime as dt
import random
from sklearn.model_selection import KFold
from tqdm import tqdm
from itertools import product

# Data Tools
import pandas as pd
import json 
import numpy as np
import torch

# Helpers
from sklearn_helpers import SKLearn_DataHelper as SKDH
from pytorch_helpers import PyTorch_DataHelper as PTDH


# Models
from CNN import CNN
import xgboost as xgb


# Performance Metrics
from carbontracker.tracker import CarbonTracker
from sklearn.metrics import mean_squared_error


class ABGM_Experiment():
    """
    This entire module entails the functionality to train both neural and non-neural models.
    It is designed to load the data efficiently, save performance, draw relevant plots and track carbon emissions. 
    """

    def __init__(self):

        self.performance_log = 'results/overall_performances.csv'
        self.performance_column_names = ['runID',
                            'model',
                            'number_of_chips',
                            'parameter_file',
                            'parameter_index',
                            'RMSE_train',
                            'RMSE_eval',
                            'RMSE_test',
                            'carbon_development',
                            'carbon_inference']
        
        


        self.model_mapping = {'CNN': CNN,
                              'ST-CNN': None}        

        self.train = {'CNN':self.train_torch_model,
                      'ST-CNN':self.train_torch_model,
                      'XGBoost':self.train_sklearn_model}


        self.train_path  = '../data/sample_files/training_data.csv'
        self.test_path  = '../data/sample_files/test_data.csv'

        self.num_folds = 5


        
    def argument_map(self):
        """
        Maps the neural and non-neural model to their respective training devices.
        Maps the number of datapoints depending on which data representation is utilized (spatial, temporal or spatio-temporal).
        """
        dp_mapping = {'CNN':12,
                   'ST-CNN':1,
                   'XGBoost':65536}

        device_mapping = {'CNN':'gpu',
                   'ST-CNN':'gpu',
                   'XGBoost':"cuda"}

        model_type_mapping = {'CNN':'neural',
                   'ST-CNN':'neural',
                   'XGBoost':"non-neural"}

        
        return device_mapping[self.model],dp_mapping[self.model]*int(self.num_dpoints),model_type_mapping[self.model]

    def load_params_to_dict(self):
        with open(self.params_file,'r') as file:
            params = json.loads(file.read())
        return params


    def handle_args(self):
        """
        Handles the commandline arguments that defines the experiment.
        """


        args = sys.argv

        if len(args) == 4:
            self.model = args[1]
            self.num_dpoints = args[2]
            self.params_file = args[3]
            self.params = self.load_params_to_dict()
            self.device,self.num_dpoints_in_representation,self.model_type = self.argument_map()

            print(f'TRAINING {self.model} ON {self.device} WITH {self.num_dpoints} DATAPOINTS USING PARAMETERS FROM {self.params_file}')
            print(f"TO THE {self.model}-MODEL {self.num_dpoints} DATAPOINTS IS EQUIVALENT TO {self.num_dpoints_in_representation} DATAPOINTS")

        else:
            print('YOU NEED TO SUPPLY 3 COMMANDLINE ARGUMENTS')
            print("="*20,'\n')
            print('MODEL [XGBoost, CNN, ST-CNN]\n')
            print('Num Training Datapoints: int {10..1000}\n')
            print('PARAMS FILE PATH [tuning_params.py,best_params.py]\n')

            sys.exit()



    def create_dirs(self):
        """
        Creates the relevant directories in order to save results and graphics
        """
        # Create the outer experiment dir
        self.experiment_start_time = dt.now().strftime("%Y_%m_%d:%H_%M_%S")
        self.experiment_dir = f"results/{self.model}/{self.experiment_start_time}/"
        os.mkdir(self.experiment_dir)

        # Create subfolders
        os.mkdir(self.experiment_dir+"plots")
        os.mkdir(self.experiment_dir+"carbon_emissions")
        os.mkdir(self.experiment_dir+"carbon_emissions/inference")
        os.mkdir(self.experiment_dir+"carbon_emissions/training")

        # Save experiment_specs
        pd.DataFrame({'model':[self.model],
                      'num_dpoints':[self.num_dpoints],
                      'params_file':[self.params_file]}).to_csv(self.experiment_dir+'experiment_specifications.csv',index=None)

        # Create the file for tracking all performances
        if os.path.exists(self.performance_log)==False:
            # Create an empty DataFrame with the specified columns
            df = pd.DataFrame(columns=self.performance_column_names)

            df.to_csv(self.performance_log,index=None)




        
    def create_kfold(self):
        """
        Function that creates kfold splits for all chipIDs and saves them globally.
        SKlearn and PyTorch runs will handle these differently.
        """

        # Create the folder where training data is saved
        self.data_path = self.experiment_dir+"training_data/"
        os.mkdir(self.data_path)
        
        # Read in training chips
        self.train_ids = pd.read_csv(self.train_path,index_col=0)
        # Get the unique ids
        unique_ids = self.train_ids['chipid'].values.tolist()

        # Shuffle
        random.shuffle(unique_ids)

        #Select the number of dpoints specified in commandline arguments
        train_ids = unique_ids[:int(self.num_dpoints)]

        self.train_ids.loc[self.train_ids['chipid'].isin(train_ids)].to_csv(self.data_path+'full_chipset.csv')

        # Create KFold splits
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=62)

        # Initialize fold dict
        self.fold2ids = {'train':{},
                    'development':{}}

        
        # assign chipIDs to train and dev for each fold
        fold = 0
        for train_index, dev_index in kf.split(train_ids):
            # Train
            self.fold2ids['train'][fold] = [train_ids[tidx] for tidx in train_index]

            # Dev
            self.fold2ids['development'][fold] = [train_ids[didx] for didx in dev_index]

            # Increment fold_index
            fold+=1


    def create_temporal_datastructure(self,datapath,cap=0):
        
        self.DataHelper = SKDH()

        dataset_dict = self.DataHelper.create_dataset(datapath)
        
        
        X,y = dataset_dict['features'],dataset_dict['target']
        
        if cap > 0:
            y[y > cap] = cap

        dataset_dict['features'],dataset_dict['target'] = None,None

        return X,y,dataset_dict
    

    def create_spatial_datastructure(self,cap=0):

        self.DataHelper = PTDH()

        dataset_dict = self.DataHelper.create_dataset(datapath)
        
        
        X,y = dataset_dict['features'],dataset_dict['target']
        
        if cap > 0:
            y[y > cap] = cap

        dataset_dict['features'],dataset_dict['target'] = None,None

        return X,y,dataset_dict
    

    def create_temporal_dataset(self):
        """
        This function uses the KFold splits to create the tabular, temporal data structure required to run an SKLearn Model.
        The data is saved in the 'training_data'-folder in the experiment directory with a sub-dir being made for each fold. 
        The test data is simply held in memory as it is (generally) much smaller than the test data, and is used more frequently.
        """

        # Create the folder where training data is saved
        fold_path = self.experiment_dir+"training_data/fold_indexes/"
        os.mkdir(fold_path)

        print('CREATING TEMPORAL TRAINING DATASTRUCTURES')

        # Iterate over all split_types (train,dev)
        for split_type,folds in tqdm(self.fold2ids.items()):

            # Iterate over all folds (0,1,2,3,4)
            for fold,indexes in folds.items():
                
                # Create flat CSV with indexes for fold so the sentinel dataset can read it
                cur_data_path = f'{fold_path}{fold}_{split_type}.csv'
                self.train_ids.loc[self.train_ids['chipid'].isin(indexes)].to_csv(cur_data_path)

                #Create temporal data_matrix for both feature data and target data
                X,y,data_dict = self.create_temporal_datastructure(cur_data_path,cap=500)
                data_dict['features'] = f"{self.data_path}{fold}/{split_type}_X.pt"
                data_dict['target'] = f"{self.data_path}{fold}/{split_type}_y.pt"

                # Create directories for the current fold
                if os.path.exists(f"{self.data_path}{fold}")==False:
                    os.mkdir(f"{self.data_path}{fold}")

                # Save the data matrices 
                torch.save(X, data_dict['features'])
                torch.save(y, data_dict['target'])

                # Dump the dictionary to a JSON file
                with open(f"{self.data_path}{fold}/{split_type}.json", 'w') as json_file:
                    json.dump(data_dict, json_file)

        del X,y,data_dict

        # Save the full dataset to be loaded for final performance
        X,y,data_dict = self.create_temporal_datastructure(cur_data_path,cap=500)

        # Save the data matrices 
        torch.save(X, f"{self.data_path}full_X.pt")
        torch.save(y, f"{self.data_path}full_y.pt")
        del X,y,data_dict

        print('CREATING THE TESTSET')
        # Create the test set but save it in memory instead of to file
        self.test_dict = self.DataHelper.create_dataset('../data/sample_files/test_data.csv')


    def create_spatial_dataset(self):
        # Create the folder where training data is saved
        fold_path = self.experiment_dir+"training_data/fold_indexes/"
        os.mkdir(fold_path)

        print('CREATING TEMPORAL TRAINING DATASTRUCTURES')

        # Iterate over all split_types (train,dev)
        for split_type,folds in tqdm(self.fold2ids.items()):

            # Iterate over all folds (0,1,2,3,4)
            for fold,indexes in folds.items():
                pass

    def create_spatio_temporal_dataset(self):
        pass

    def create_data_representations(self):
        # Uses the fold2chipids to create the training, evaluation and test data in the correct representations
        
        # Ikke Neurale Metoder
        ## XGBboost
        #### use the chipIDs to load all datapoints and create k 2D training files and save them
        #### When training simply load in each fold as a numpy matrix
        #### Do the equivalent to the test data

        if self.model == 'XGBoost':
            self.create_temporal_dataset()
        elif self.model == 'CNN':
            self.create_spatial_dataset()
        elif self.model == 'ST-CNN':
            self.create_spatio_temporal_dataset()

        # Neurale Metoder
        # TODO
        ## CNN
        #### Konstruer en liste af alle (256x256x15) datapunkter
        #### Konstruer en liste af alle (256x256) targets


        ## ST-CNN
        #### Konstruer en liste af alle (256x256x15x12) datapunkter
        #### Konstruer en liste af alle (256x256) targets 



        ## Konstruer en dataloader ud fra træningsdata og targets




    
    def create_dataset(self):
        """
        Creates an efficient way to train the model 
        """

        self.create_dirs()

        self.create_kfold()

        self.create_data_representations()
        
        # Depends on 
        #### self.model
        #### self.num_dpoints
        
        #1 Create folds with chip IDs (general)
        #2 Either
        #### 2a Create neural dataloaders
        #### 2b Create non-neural flat_data_files

    def create_param_permutations(self):
        print('CREATING PARAMETER LIST')
        
        param_permutations = []
        # Get the keys and values from the dictionary
        keys = self.params.keys()
        values = self.params.values()

        # Create a list of dictionaries with all possible combinations
        param_combinations = [dict(zip(keys, combination)) for combination in product(*values)]

        # Print the list of dictionaries
        for params in param_combinations:
            param_permutations.append(params)


        return param_permutations
    
    def evaluateXGB(self,xgb_instance,deval,eval_target,verbose):

        self.predictions = xgb_instance.predict(deval)

        rmse = np.sqrt(mean_squared_error(eval_target,self.predictions))
        
        if verbose:
            return rmse, self.predictions
        else:
            return rmse
            

    def train_sklearn_model(self):
        print('Initiating Training')

        # Set device in params
        
        
        # Create all possible parameter conficgurations
        param_permutations = self.create_param_permutations()


        # Setup CarbonTracker
        tracker = CarbonTracker(epochs=1,
                                log_dir=f"{self.experiment_dir}/carbon_emissions/training",
                                devices_by_pid=True)
        
        tracker.epoch_start()
        
        # Instantiate datastructure for performances on each fold for each parameter configuration
        param_performances = [[]]*len(param_permutations)
        print('TRAINING LOOP')
        for fold in tqdm(range(self.num_folds)):
            # Load training data for fold
            X_train = torch.load(f"{self.experiment_dir}training_data/{fold}/train_X.pt")
            y_train = torch.load(f"{self.experiment_dir}training_data/{fold}/train_y.pt")

            print('TRAINING ON DATASIZE OF DIMENSIONALITY',X_train.shape)

            dtrain = xgb.DMatrix(X_train,y_train)

            # Load developmentdata for fold
            X_dev = torch.load(f"{self.experiment_dir}training_data/{fold}/development_X.pt")
            y_dev = torch.load(f"{self.experiment_dir}training_data/{fold}/development_y.pt")
            ddev = xgb.DMatrix(X_dev,y_dev)

            for idx,params in tqdm(enumerate(param_permutations)):
                params['device'] = self.device

                # Train Model
                xgb_instance = xgb.train(params,dtrain)

                # Evaluate the performance
                RMSE = self.evaluateXGB(xgb_instance,ddev,y_dev,verbose=False)
                
                # Append the fold performance for the config to the datastructure
                param_performances[idx].append(RMSE)

        tracker.epoch_end()

        # Free Up Memory
        del X_train,y_train,X_dev,y_dev


        # Average over folds for each parameter setting getting the mean RMSE for the config
        aggregated_performances = [np.mean(param_folds) for param_folds in param_performances]
        
        # Get the index of the best setting
        best_param_index = np.argmin(aggregated_performances)
        
        # Get the best training performance on the dev set
        self.train_RMSE = aggregated_performances[best_param_index]

        # Get the parameter setting yielding the best dev performance
        best_params = param_permutations[best_param_index]
        self.best_params = best_params

        # Setup CarbonTracker for inference
        tracker = CarbonTracker(epochs=1,
                                log_dir=f"{self.experiment_dir}/carbon_emissions/inference",
                                devices_by_pid=True)
        
        tracker.epoch_start()
        # Load the full train set
        X_full = torch.load(f"{self.data_path}full_X.pt")
        y_full = torch.load(f"{self.data_path}full_y.pt")
        dfull = xgb.DMatrix(X_full,y_full)

        # Fit the model to the full trainset
        best_xgb = xgb.train(best_params,dfull)

        # Create XGB Data Matrix from testdata
        dtest = xgb.DMatrix(self.test_dict['features'],
                            self.test_dict['target'])

        
        # Evaluate final model on test data
        self.test_RMSE,predictions =self.evaluateXGB(xgb_instance,dtest,self.test_dict['target'],verbose=True)


        # End Inference carbion tracking 
        tracker.epoch_end()

        # Save predictions along with ground truth data
        pd.DataFrame({'predicted':predictions,
                      'ground_truth':self.test_dict['target'].numpy()})\
                      .to_csv(self.experiment_dir+'estimates.csv')


        



    def train_torch_model(self):
        print('Initiating Training')

        # Set device in params
        
        
        # Create all possible parameter conficgurations
        param_permutations = self.create_param_permutations()


        # Setup CarbonTracker
        tracker = CarbonTracker(epochs=1,
                                log_dir=f"{self.experiment_dir}/carbon_emissions/training",
                                devices_by_pid=True)
        
        tracker.epoch_start()
        
        # Instantiate datastructure for performances on each fold for each parameter configuration
        param_performances = [[]]*len(param_permutations)
        print('TRAINING LOOP')
        for fold in tqdm(range(self.num_folds)):
            # Load training data for fold
            X_train = torch.load(f"{self.experiment_dir}training_data/{fold}/train_X.pt")
            y_train = torch.load(f"{self.experiment_dir}training_data/{fold}/train_y.pt")




    def visualize(self):
        """
        Small function for running the two possible visualization cases.
        One for tabular cases, where the original data needs to be reconstructed and one for 2D cases. 
        """
        if self.model == 'XGBoost':
            # Get the chip_dict for plotting
            chip_dict = self.DataHelper.find_best_and_worst_chip(self.experiment_dir)

            # Create all relevant plots
            self.DataHelper.run_all_plots(chip_dict,self.test_dict,self.experiment_dir)

        else:
            pass


    def save_results(self):
        """
        Saves the performance of the model_run to the global lookup table
        """
        row = {'runID':[self.experiment_start_time],
                'model':[self.model],
                'number_of_chips':[self.num_dpoints],
                'parameters':[self.best_params],
                'RMSE_train':[self.train_RMSE],
                'RMSE_test':[self.test_RMSE],
                'carbon_development':[0],
                'carbon_inference':[0]}

        pd.DataFrame(row).to_csv('results/overall_performances.csv',header=False,index=None,mode='a',sep=';')

        


    def run_experiment(self):
        """
        Runs the entire experiment according to the arguments and parameters supplied in the commandline
        """
        # Handle the commandline arguments defining the experiment
        self.handle_args()

        # Create the training,evaluation and testing dataset, as well as all relevant directories
        self.create_dataset()

        # Train Model
        self.train[self.model]()

        # Save results to global dataframe
        self.save_results()

        self.visualize()

        





            


if __name__ =='__main__':

    Experiment = ABGM_Experiment()
    
    Experiment.run_experiment()
    
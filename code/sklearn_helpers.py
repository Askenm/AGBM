import numpy as np
import pandas as pd
import rasterio
import torch
import transforms as tf
import data_loader as dl
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import os
import visualisation as viz

class SKLearn_DataHelper():
    def __init__(self):
        pass
    
    def load_SentinelChipset(self,
                fpath="test_sample.csv",
                max_chips=None,
                dir_tiles="../data/train_features/",
                dir_target="../data/train_agbm/",
                ):

        """
        This function takes a pandas dataframe with chipIDs and creates a custom dataset instance from the actual chip data.
        """
        
        dataset = dl.SentinelDataset(
            tile_file=fpath,
            dir_tiles=dir_tiles,
            dir_target=dir_target,
            max_chips=max_chips,
            transform=None
        )

        return dataset



    def singletile2tabular(self,yearly_tile,yearly_target):
        rs_full = []

        for ts in yearly_tile:
            rs_new = ts.reshape(15,-1).T
            
            rs_full.append(rs_new)
            
        reshaped_tile = torch.concat(rs_full,dim=1)#.numpy()

        reshaped_target = yearly_target.reshape(-1)#.numpy()

        return reshaped_tile,reshaped_target



    def convert2tabular(self,input_,target_):

        tiles = []
        targets = []

        input_len = len(input_)
        
        for yearly_tile_idx in tqdm(range(0,input_len,12)):
            
            rs_tile,rs_target = self.singletile2tabular(input_[yearly_tile_idx:yearly_tile_idx+12],target_[yearly_tile_idx//12])
            tiles.append(rs_tile)
            targets.append(rs_target)
        
        full_featureset = torch.concat(tiles,dim=0)

        full_targetset = torch.concat(targets,dim=0)

        return full_featureset,full_targetset



    def create_dataset(self,chip_filepath):
            
        dataset = self.load_SentinelChipset(chip_filepath)

        input_ = []
        target_ = []
        tiles = dataset.df_tile_list.chipid.unique().tolist()

    
        for i in tqdm(range(len(dataset))):
            all_channels = [channel for channel in dataset.__getitem__(i)["image"]]

            

            input_.append(torch.stack(all_channels))

            if i%12 == 0:
                target_.append(dataset.__getitem__(i)["label"])
    
        input2D,target1D = self.convert2tabular(input_,target_)
    
        return {'features':input2D,'target':target1D,'tile_ids':tiles}



    def find_best_and_worst_chip(self,experiment_dir):
        fpath = experiment_dir+'estimates.csv'
        estimates = pd.read_csv(fpath)


        stepsize = 256**2

        num_steps = estimates.shape[0]//stepsize
        
        performances = []
        
        
        for step in tqdm(range(num_steps)):
            data_point = estimates.iloc[stepsize*step:stepsize*(step+1)]
            y_true = data_point['ground_truth'].values
            y_pred = data_point['predicted'].values
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            
            performances.append(rmse)
            
            
        
        best_chip_perf,best_chip_idx = min(performances),np.argmin(performances)
        worst_chip_perf,worst_chip_idx = max(performances),np.argmax(performances)

        best = estimates.iloc[stepsize*best_chip_idx:stepsize*(best_chip_idx+1)]
        worst = estimates.iloc[stepsize*worst_chip_idx:stepsize*(worst_chip_idx+1)]

        return {"best_chip_perf":best_chip_perf,"best":best,"best_chip_idx":best_chip_idx,
                "worst_chip_perf":worst_chip_perf,"worst":worst,"worst_chip_idx":worst_chip_idx}
        

    def run_all_plots(self,chip_dict,test_data,experiment_dir):
        # Define the stepsize to recreate the images from the flattened data
        stepsize = 256**2

        # get the starting and ending indexes for the best performing chip
        a_best = chip_dict['best_chip_idx']*stepsize
        b_best = (1+chip_dict['best_chip_idx'])*(stepsize)

        # get the starting and ending indexes for the worst performing chip
        a_worst = chip_dict['worst_chip_idx']*stepsize
        b_worst = (1+chip_dict['worst_chip_idx'])*(stepsize)


        # Get the best and word chips
        best_target = test_data['target'][a_best:b_best]
        worst_target = test_data['target'][a_worst:b_worst]


        best_target_estimate = chip_dict['best']['predicted'].values
        worst_target_estimate = chip_dict['worst']['predicted'].values

        
        # Get the chipID for the best and worst chip (only used for plot titles)
        best_chip_id = test_data['tile_ids'][chip_dict['best_chip_idx']]
        worst_chip_id = test_data['tile_ids'][chip_dict['worst_chip_idx']]

        
        # 3D PLOTS
        print('COMMENCING BEST PLOTS')
        
        # Create the relevant plots
        best_dir = experiment_dir+'plots/best/'
        os.mkdir(best_dir)

        _max = best_target.max()

        # 3D Plots
        viz.plot_target_chip(best_target,best_chip_id+f" (RMSE: {chip_dict['best_chip_perf']})",z_max = _max,estimate=None,savedir=best_dir+'target.png')  
        viz.plot_target_chip(best_target_estimate,best_chip_id+" (estimate)"+f" (RMSE: {chip_dict['best_chip_perf']})",z_max = _max,estimate=None,savedir=best_dir+'estimate.png')  
        viz.plot_target_chip(best_target,f"{best_chip_id} Error heatmap (Estimate)"+f" (RMSE: {chip_dict['best_chip_perf']})",z_max = _max,estimate=best_target_estimate,savedir=best_dir+'heat_map_estimate.png')  
        viz.plot_target_chip(best_target,f"{best_chip_id} Error heatmap (Target)"+f" (RMSE: {chip_dict['best_chip_perf']})",z_max = _max,estimate=best_target_estimate,savedir=best_dir+'heat_map_target.png')  

        # 2D Plots
        viz.plot_performance_scatter(df=chip_dict['best'],
                                     rmse=chip_dict['best_chip_perf'],
                                     savedir=best_dir)


        print('COMMENCING WORST PLOTS')
        worst_dir = experiment_dir+'plots/worst/'
        os.mkdir(worst_dir)
        

        _max = worst_target.max()
        # 3D Plots
        viz.plot_target_chip(worst_target,worst_chip_id+f" (RMSE: {chip_dict['worst_chip_perf']})",z_max = _max,estimate=None,savedir=worst_dir+'target.png')  
        viz.plot_target_chip(worst_target_estimate,worst_chip_id+" (estimate)"+f" (RMSE: {chip_dict['worst_chip_perf']})",z_max = _max,estimate=None,savedir=worst_dir+'estimate.png')  
        viz.plot_target_chip(worst_target_estimate,f"{worst_chip_id} Error heatmap (Estimate)"+f" (RMSE: {chip_dict['worst_chip_perf']})",z_max = _max,estimate=worst_target,savedir=worst_dir+'heat_map_estimate.png')  
        viz.plot_target_chip(worst_target,f"{worst_chip_id} Error heatmap (Target)"+f" (RMSE: {chip_dict['worst_chip_perf']})",z_max = _max,estimate=worst_target_estimate,savedir=worst_dir+'heat_map_target.png')  


        # 2D PLots
        viz.plot_performance_scatter(df=chip_dict['worst'],
                                     rmse=chip_dict['worst_chip_perf'],
                                     savedir=worst_dir)



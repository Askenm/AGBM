import numpy as np
import rasterio
import torch
import transforms as tf
import data_loader as dl
from tqdm import tqdm
import pandas as pd
import os


class PyTorch_DataHelper():
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



    def create_dataset(self,chip_filepath):
            
        dataset = self.load_SentinelChipset(chip_filepath)

        input_ = []
        target_ = []
        tiles = dataset.df_tile_list.chipid.unique().tolist()

    
        for i in tqdm(range(len(dataset))):

            all_channels = [channel for channel in dataset.__getitem__(i)["image"]]

            input_.append(torch.stack(all_channels))
            target_.append(dataset.__getitem__(i)["label"])

        print(len(input_),input_[0].shape)
        print(len(target_),target_[0].shape)
        #return {'features':input2D,'target':target1D,'tile_ids':tiles}
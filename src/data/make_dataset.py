# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()


import torch
from torch.utils.data import Dataset
import os
import wget
import numpy as np
import pickle

class CorruptMnist(Dataset):
    def __init__(self,train):
        self.download_data(train)
        if train:
            content = [ ]
            for i in range(5):
                content.append(np.load(f"train_{i}.npz", allow_pickle=True))
            data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
        else:
            content = np.load("test.npz", allow_pickle=True)
            data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['labels'])
        
        # Normalize data
        mean = data.mean(dim=(1,2,3), keepdim=True)
        std = data.std(dim=(1,2,3), keepdim=True)

        data = (data - mean) / std

        self.data = data
        self.targets = targets

    def download_data(self,train):
        files=os.listdir()
        for file_idx in range(5):
                        if f'train_{file_idx}.npz' not in files:
                            wget.download(f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/train_{file_idx}.npz")

        if "test.npz" not in files:    
                        wget.download("https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/test.npz")

    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]

if __name__ == "__main__":
    cwd=os.getcwd()
    os.chdir("./data/raw")
    dataset_train = CorruptMnist(train=True)
    dataset_test = CorruptMnist(train=False)

    os.chdir("../processed")
    
    # Open a file for writing
    with open('dataset_train.pt', 'wb') as f:
    # Serialize the object and write it to the file
        pickle.dump(dataset_train, f)

    # Open a file for writing
    with open('dataset_test.pt', 'wb') as f:
    # Serialize the object and write it to the file
        pickle.dump(dataset_test, f)

    os.chdir(cwd)






'''
protein dataset
'''
import os
import copy
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Dataset


class Protein_Dataset(Dataset):
    def __init__(self, root, test_mode=False, pre_filter=None, pre_transform=None):
        '''init func

        Input:
            - root(str):test data path 
            - test_mode(str):train or test mode
            - pre_filter:filter process
            - pre_transform:transform process
        '''
        self.raw_file_path = os.path.join(root, 'raw')
        self.processed_file_path = os.path.join(root, 'processed')
        self.root = root
        self.test_mode = test_mode
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        super(Protein_Dataset, self).__init__(root, pre_filter, pre_transform)

        if not os.path.exists(self.processed_file_path) or len(os.listdir(self.processed_file_path)) < self.len():
            self.data_num_features, self.data_num_classes = self.process()
        else:
            if not self.test_mode:
                self.data_num_features = self.get(0).x.shape[1]
                self.data_num_classes = self.get(0).pos.shape[1]
            else:
                self.data_num_features = self.get(0).x.shape[1]
                self.data_num_classes = 3

    @property
    def raw_file_names(self):
        '''read raw file names

        Output:
            - file_path_list(list):raw file list
        '''
        file_name_list = os.listdir(self.raw_file_path)
        file_path_list = []
        for name in file_name_list:
            file_path_list.append(os.path.join(self.raw_file_path, name))

        return file_path_list

    @property
    def processed_file_names(self):
        '''generate processed file names

        Output:
            - data_pt_list(list):processed file list
        '''
        if not os.path.exists(self.processed_file_path):
            os.mkdir(self.processed_file_path)
        data_pt_list = []
        for i in range(len(os.listdir(self.raw_file_path))):
            data_pt_list.append('data_{}.pt'.format(i))
        return data_pt_list

    def download(self):
        # Download to `self.raw_dir`.
        pass


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_file_path, 'data_{}.pt'.format(idx)))
        return data




from .video_record import VideoRecord
from torch.utils.data import Dataset
import os
import torchaudio.transforms as T
import pandas as pd
import torch
import numpy as np
import torch.utils.data as data
from abc import ABC
from utils.logger import logger
import math

#convert labels from string to integer
activities_to_classify = {
    'Get/replace items from refrigerator/cabinets/drawers': 0,
    'Peel a cucumber': 1,
    'Clear cutting board': 2,
    'Slice a cucumber': 3,
    'Peel a potato': 4,
    'Slice a potato': 5,
    'Slice bread': 6,
    'Spread almond butter on a bread slice': 7,
    'Spread jelly on a bread slice': 8,
    'Open/close a jar of almond butter': 9,
    'Pour water from a pitcher into a glass': 10,
    'Clean a plate with a sponge': 11,
    'Clean a plate with a towel': 12,
    'Clean a pan with a sponge': 13,
    'Clean a pan with a towel': 14,
    'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 15,
    'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 16,
    'Stack on table: 3 each large/small plates, bowls': 17,
    'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 18,
    'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 19,
}

class ActionNetDataset(data.Dataset, ABC):
    def __init__(self, directory, filename):
        self.df = pd.read_pickle(os.path.join(directory, filename))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        emg, label = self.df.loc[idx, 'emg_data'],self.df.loc[idx,'description']
       
        if label== 'Get items from refrigerator/cabinets/drawers' or label== 'Replace items from refrigerator/cabinets/drawers' :
            emg = torch.tensor(emg, dtype=torch.float32)
            label = torch.tensor(0)
            emg = {"EMG": emg.unsqueeze(0)} 
            return emg,label 
        elif label=='Open a jar of almond butter' or label=='Close a jar of almond butter':
            emg = torch.tensor(emg, dtype=torch.float32)
            label = torch.tensor(9)
            emg = {"EMG": emg.unsqueeze(0)} 
            return emg,label 
           
        emg = torch.tensor(emg, dtype=torch.float32)
        label = torch.tensor(activities_to_classify.index(label))
        emg = {"EMG": emg.unsqueeze(0)} 
        return emg,label
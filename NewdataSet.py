import torch
from torch.utils.data import Dataset


class dataset_prediction(Dataset):
    def __init__(self, data_features, data_target):
        self.len = len(data_features)
        self.features = torch.from_numpy(data_features)
        self.target = torch.from_numpy(data_target).to(torch.long)

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len

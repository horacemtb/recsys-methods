import numpy as np
import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    """
    Pytorch dataset for movieLens interactions. Expects a df with columns: user, item, label
    """
    def __init__(self, interactions):
        self.users = interactions['user'].values
        self.items = interactions['item'].values
        self.labels = interactions['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.LongTensor([self.users[idx]]),
                torch.LongTensor([self.items[idx]]),
                torch.FloatTensor([self.labels[idx]])
                )

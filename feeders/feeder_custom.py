import pickle
import numpy as np
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, data_path, label_path, debug=False):
        self.data_path = data_path
        self.label_path = label_path
        self.debug = debug
        self.load_data()

    def load_data(self):
        self.data = np.load(self.data_path)

        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            self.sample_name = self.sample_name[:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [self.label[i] in rank[i, -top_k:] for i in range(len(self.label))]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
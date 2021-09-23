import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class FlickrDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        # columns = [image, caption]
        self.csv = self.load_dataset(csv_path)

    def load_dataset(self, csv_path):
        csv = pd.read_csv(csv_path)
        return csv


class FlickrDataLoader(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        self.dataset = FlickrDataset(csv_path)

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.dataset)


def main():
    train_loader = FlickrDataLoader(csv_path='data/captions.csv')


if __name__ == '__main__':
    print(1)
    main()

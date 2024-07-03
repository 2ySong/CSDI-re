import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch


class Forecasting_Dataset(Dataset):
    def __init__(self, datatype, mode="train"):

        self.history_length = 168
        self.pred_length = 24

        if datatype == "electricity":
            datafolder = "./datasets/electricity_nips"
            self.test_length = 24 * 7  # 192
            self.valid_length = 24 * 5  # 120

        self.seq_length = self.history_length + self.pred_length  # 168+24

        paths = datafolder + "/data.pkl"
        # shape: (T x N)
        # mask_data is usually filled by 1
        with open(paths, "rb") as f:
            self.main_data, self.mask_data = pickle.load(f)

        paths = datafolder + "/meanstd.pkl"
        with open(paths, "rb") as f:
            self.mean_data, self.std_data = pickle.load(f)

        self.main_data = (self.main_data - self.mean_data) / self.std_data

        total_length = (len(self.main_data) - 1)
        # print(total_length)
        if mode == "train":
            start = 0
            end = (
                    total_length
                    - self.seq_length
                    - self.valid_length
                    - self.test_length
                    + 1
            )  # T-192-120-192+1
            self.use_index = np.arange(start, end, 1)

            # print(start,"->", end,",")
        if mode == "valid":  # valid
            start = (
                    total_length
                    - self.seq_length
                    - self.valid_length
                    - self.test_length
                    + self.pred_length
            )
            end = total_length - self.seq_length - self.test_length + self.pred_length
            self.use_index = np.arange(start, end, self.pred_length)#第三个字段表示步长
            # print(start,"->", end)
        if mode == "test":  # test
            start = total_length - self.seq_length - self.test_length + self.pred_length
            end = total_length - self.seq_length + self.pred_length
            self.use_index = np.arange(start, end, self.pred_length)
            # print(start,"->", end)

    def __getitem__(self, orgindex):
        index = int(self.use_index[orgindex])
        # target_mask = self.mask_data[index: index + self.seq_length].copy()-szy
        target_mask = self.mask_data[index: index + self.seq_length].copy()
        target_mask[-self.pred_length:] = 0.0  # pred mask for test pattern strategy
        s = {
            "observed_data": self.main_data[index: index + self.seq_length],
            "observed_mask": self.mask_data[index: index + self.seq_length],
            "gt_mask": target_mask,
            "timepoints": np.arange(self.seq_length) * 1.0,
            "feature_id": np.arange(self.main_data.shape[1]) * 1.0,
        }

        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader(datatype, device, batch_size=8):
    dataset = Forecasting_Dataset(datatype, mode="train")
    train_len = int(len(dataset))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = Forecasting_Dataset(datatype, mode="valid")
    valid_len = int(len(valid_dataset))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = Forecasting_Dataset(datatype, mode="test")
    test_len = int(len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    scaler = torch.from_numpy(dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    lengths = [train_len, valid_len, test_len]

    return train_loader, valid_loader, test_loader, scaler, mean_scaler, lengths

import os
import re
import numpy as np
from scipy import signal
from tqdm import tqdm
from torch.utils.data import Dataset
from data_processing import split_by_suffix_digit

def dataProcessing_3(
    file_path,
    length=5120,
    step_size=5120,
):
    filenames = os.listdir(file_path)

    train_files, val_files, test_files = split_by_suffix_digit(filenames)


    def capture(files):
        data = {}
        for f in tqdm(files):
            path = os.path.join(file_path, f)
            data[f] = np.genfromtxt(path, delimiter=",", dtype=float,skip_header=1, max_rows=122880)
        return data

    def parse_label(filename):
        match = re.search(r'电流-([A-Z+]+)-S', filename)
        key_mapping = {
            'A': 0, 'C': 1, 'G': 2, 'N': 3,
            'B': 4, 'B+G': 5, 'H': 6, 'H+N': 7, 'E': 8
        }
        return key_mapping.get(match.group(1), -1) if match else -1

    def process_and_slice(data_dict):
        Samples, Labels = [], []

        for fname, raw in tqdm(data_dict.items()):
            label = parse_label(fname)

            resample_data = signal.resample(raw, len(raw) // 4)

            reference = resample_data[:512]
            baseline = np.mean(reference)

            num_blocks = len(resample_data) // 512
            for i in range(1, num_blocks):
                block = resample_data[i * 512:(i + 1) * 512]
                resample_data[i * 512:(i + 1) * 512] -= (np.mean(block) - baseline)

            min_val, max_val = resample_data.min(), resample_data.max()
            if min_val != max_val:
                resample_data = (resample_data - min_val) / (max_val - min_val)

            num_samples = (len(resample_data) - length) // step_size + 1
            for i in range(num_samples):
                sample = resample_data[i * step_size: i * step_size + length]
                if len(sample) == length:
                    Samples.append([sample] * 4)
                    Labels.append([label] * 4)

        return np.array(Samples), np.array(Labels)

    train_data = capture(train_files)
    val_data   = capture(val_files)
    test_data  = capture(test_files)

    Train_X, Train_Y = process_and_slice(train_data)
    Val_X,   Val_Y   = process_and_slice(val_data)
    Test_X,  Test_Y  = process_and_slice(test_data)

    return Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y


class CustomTensorDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        return x, self.y[index]

    def __len__(self):
        return len(self.X)

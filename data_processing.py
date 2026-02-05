import os
import re
import numpy as np
from tqdm import tqdm

def split_by_suffix_digit(filenames):
    train_files, val_files, test_files = [], [], []

    for f in filenames:
        f = str(f)
        m = re.search(r'_(\d+)\.csv$', f)
        if not m:
            continue

        idx = m.group(1)

        if idx == '4':
            val_files.append(f)
        elif idx == '5':
            test_files.append(f)
        else:
            train_files.append(f)

    return train_files, val_files, test_files

def dataProcessing(file_path, step_size=5120):

    filenames = np.array(os.listdir(file_path))

    train_files, val_files, test_files = split_by_suffix_digit(filenames)

    def capture(path, file_list):
        P, V = {}, {}
        for fname in tqdm(file_list):
            data = np.genfromtxt(
                os.path.join(path, fname),
                delimiter=",",
                dtype=float,
                skip_header=1,
                max_rows=122880
            )
            P[fname] = data[:, 0:1]
            V[fname] = data[:, 1:]
        return P, V

    def parse_label(filename):
        match = re.search(r'电流-([A-Z+]+)-S', filename)
        key_mapping = {
            'A': 0, 'C': 1, 'G': 2, 'N': 3,
            'B': 4, 'B+G': 5, 'H': 6, 'H+N': 7, 'E': 8
        }
        return key_mapping.get(match.group(1), -1) if match else -1

    def slice_files(P_dict, V_dict):
        Data_P, Data_V, Labels = [], [], []

        for fname in tqdm(P_dict.keys()):
            p = P_dict[fname]
            v = V_dict[fname]
            label = parse_label(fname)

            num_seg = p.shape[0] // step_size
            for i in range(num_seg):
                start = i * step_size
                p_seg = p[start:start + step_size]
                v_seg = v[start:start + step_size]

                if p_seg.shape[0] == step_size and v_seg.shape[0] == step_size:
                    Data_P.append(p_seg)
                    Data_V.append(v_seg)
                    Labels.append(label)

        return (
            np.array(Data_P),
            np.array(Data_V),
            np.array(Labels)
        )

    Train_P_raw, Train_V_raw = capture(file_path, train_files)
    Val_P_raw,   Val_V_raw   = capture(file_path, val_files)
    Test_P_raw,  Test_V_raw  = capture(file_path, test_files)

    Train_P, Train_V, Train_Y = slice_files(Train_P_raw, Train_V_raw)
    Val_P,   Val_V,   Val_Y   = slice_files(Val_P_raw,   Val_V_raw)
    Test_P,  Test_V,  Test_Y  = slice_files(Test_P_raw,  Test_V_raw)

    return Train_P, Train_V, Train_Y,Val_P,   Val_V,   Val_Y, Test_P,  Test_V,  Test_Y



def standardize_with_train(train_data, val_data,test_data):
    s = np.array([0,0,0])
    s1 = np.broadcast_to(s, (5120,3))
    valid_train_mask = np.stack([np.all(item == s1, axis=(0, 1)) for item in train_data])
    valid_train_mask = ~valid_train_mask
    valid_train = train_data[valid_train_mask]


    mean = valid_train.mean(axis=(0, 1), keepdims=True)
    std = valid_train.std(axis=(0, 1), keepdims=True)

    def standardize(data):
        data_std = data.copy()
        valid_mask = np.stack([np.all(item == s1, axis=(0, 1)) for item in data])
        valid_mask = ~valid_mask
        data_std[valid_mask] = (data[valid_mask] - mean) / std
        return data_std


    train_data_std = standardize(train_data)
    val_data_std = standardize(val_data)
    test_data_std = standardize(test_data)

    return train_data_std,val_data_std, test_data_std

def collect_data1(batch):

    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    y = np.array(labels)
    data = np.array(data)

    S_P = data[:, :, 0:1]
    S_V = data[:, :, 1:4]
    S_P1= data[:, :, 4:5]

    s = np.array([0,0,0])
    s1 = np.broadcast_to(s, (5120, 3))


    is_zero = np.stack([np.all(item == s1, axis=(0,1)) for item in S_V])

    pairs = ~is_zero

    return [y, S_P, S_V, pairs,S_P1]

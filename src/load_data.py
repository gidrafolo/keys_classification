import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

DATA_FOLDER = ('..\\data')

def read_and_process_data(data_folder):
    data = []
    labels = []

    for key_folder in os.listdir(data_folder):
        key_path = os.path.join(data_folder, key_folder)
        key_index = int(key_folder.split('_')[-1])

        for file in os.listdir(key_path):
            file_path = os.path.join(key_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                times_and_pressures = [line.strip().split('\t') for line in lines[8:]]

                pressures = [float(tp[1]) for tp in times_and_pressures if len(tp) == 2]
                pressures = (pressures - np.mean(pressures)) / np.std(pressures)

                data.append(pressures)
                labels.append(key_index)

    return np.array(data), np.array(labels)



def get_data_loaders():
    data, labels = read_and_process_data(DATA_FOLDER)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    train_data = TensorDataset(torch.tensor(X_train[:, None, :], dtype=torch.float32),
                               torch.tensor(y_train, dtype=torch.int64))
    test_data = TensorDataset(torch.tensor(X_test[:, None, :], dtype=torch.float32),
                              torch.tensor(y_test, dtype=torch.int64))
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    return (train_loader, test_loader)


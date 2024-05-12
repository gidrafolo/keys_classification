import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class KeysClassifier(nn.Module):
    def __init__(self):
        super(KeysClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=2, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=2, padding=1)
        self.dropout = nn.Dropout(0.8)
        self.fc1 = nn.Linear(16 * 2500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Модель переобучается, она слишком сложна
# class KeysClassifier(nn.Module):
#     def __init__(self):
#         super(SoundClassifier, self).__init__()
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(64 * 1250, 100)
#         self.fc2 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.pool(torch.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
import torch
import numpy as np

from src.model import KeysClassifier
from scripts.train import PATH_TO_SAVE_MODEL

def load_model(model_path):
    model = KeysClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = [line.strip().split('\t')[1] for line in lines[8:]]
        pressures = np.array(data, dtype=np.float32)
        pressures = (pressures - np.mean(pressures)) / np.std(pressures)
    return torch.tensor(pressures[None, None, :])


def model_predict(path_to_model, file_path):
    model = load_model(path_to_model)
    data = preprocess_data(file_path)
    output = model(data)
    prediction = torch.argmax(output, dim=1)
    return prediction.item()


if __name__ == '__main__':
    file_path = '..\\data\\key_7\\Измерение 13.dtu'
    prediction = model_predict(file_path)
    print(f'Predicted class: {prediction}')
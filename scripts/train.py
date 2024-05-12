import torch
import torch.nn as nn
import torch.optim as optim

from src.model import KeysClassifier
from src.load_data import get_data_loaders

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report


PATH_TO_SAVE_MODEL = '..\\models\\KeysClassifier.pth'

def main():
    model = KeysClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data_loaders()
    train_losses, train_accuracies, test_accuracies = train_and_evaluate_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer
    )
    plot_and_save_metrics(train_losses, train_accuracies, test_accuracies)
    torch.save(model.state_dict(), PATH_TO_SAVE_MODEL)

def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=25):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = train_correct / total_train

        test_accuracy, report = evaluate_model(model, test_loader)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}, Train Acc: {train_accuracy*100:.2f}%, Test Acc: {test_accuracy*100:.2f}%')
        print('Test Classification Report:')
        print(report)
        print('-------------------------=================-------------------')


    return(train_losses, train_accuracies, test_accuracies)

def evaluate_model(model, test_loader):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())
            true_labels.extend(labels.numpy())


    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=[f'Key {i}' for i in range(10)])

    return accuracy, report


def plot_and_save_metrics(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == '__main__':
    main()
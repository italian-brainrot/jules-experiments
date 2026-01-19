import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import mnist1d.data as mnist1d_data
from light_dataloader import TensorDataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define the model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def get_data():
    args = mnist1d_data.get_dataset_args()
    data = mnist1d_data.get_dataset(args, path='./mnist1d_data.pkl')

    x_train = torch.from_numpy(data['x']).float()
    y_train = torch.from_numpy(data['y']).long()
    x_test = torch.from_numpy(data['x_test']).float()
    y_test = torch.from_numpy(data['y_test']).long()

    return x_train, y_train, x_test, y_test

class ProgressiveBatchScheduler:
    def __init__(self, data, initial_batch_size, max_batch_size, increment):
        self.data = data
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.increment = increment
        self.num_samples = len(self.data[0])
        self.indices = list(range(self.num_samples))

    def __iter__(self):
        self.current_pos = 0
        self.batch_size = self.initial_batch_size
        np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_pos >= self.num_samples:
            raise StopIteration

        end_pos = self.current_pos + self.batch_size
        batch_indices = self.indices[self.current_pos:end_pos]

        batch_x = self.data[0][batch_indices]
        batch_y = self.data[1][batch_indices]

        self.current_pos = end_pos
        self.batch_size = min(self.max_batch_size, self.batch_size + self.increment)

        return batch_x, batch_y

    def __len__(self):
        # Approximation of the number of batches
        avg_batch_size = (self.initial_batch_size + self.max_batch_size) / 2
        return int(np.ceil(self.num_samples / avg_batch_size))


def train(model, train_loader, optimizer, criterion):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.data[0])
    accuracy = 100. * correct / len(test_loader.data[0])
    return test_loss, accuracy


def objective(trial, x_train, y_train, x_val, y_val, strategy):
    input_size = 40
    hidden_size = 256
    output_size = 10
    epochs = 20

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = MLP(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if strategy == 'small_fixed':
        train_loader = TensorDataLoader((x_train, y_train), batch_size=32, shuffle=True)
    elif strategy == 'large_fixed':
        train_loader = TensorDataLoader((x_train, y_train), batch_size=512, shuffle=True)
    elif strategy == 'progressive':
        train_loader = ProgressiveBatchScheduler((x_train, y_train), initial_batch_size=32, max_batch_size=512, increment=1)

    val_loader = TensorDataLoader((x_val, y_val), batch_size=512, shuffle=False)

    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion)

    val_loss, _ = evaluate(model, val_loader, criterion)
    return val_loss

def main():
    x_train, y_train, x_test, y_test = get_data()

    strategies = ['small_fixed', 'large_fixed', 'progressive']
    results = {}

    for strategy in strategies:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, strategy), n_trials=20)

        best_lr = study.best_params['lr']

        # Train the final model with the best learning rate
        input_size = 40
        hidden_size = 256
        output_size = 10
        epochs = 100

        model = MLP(input_size, hidden_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=best_lr)
        criterion = nn.CrossEntropyLoss()

        if strategy == 'small_fixed':
            train_loader = TensorDataLoader((x_train, y_train), batch_size=32, shuffle=True)
        elif strategy == 'large_fixed':
            train_loader = TensorDataLoader((x_train, y_train), batch_size=512, shuffle=True)
        elif strategy == 'progressive':
            train_loader = ProgressiveBatchScheduler((x_train, y_train), initial_batch_size=32, max_batch_size=512, increment=1)

        val_loader = TensorDataLoader((x_test, y_test), batch_size=512, shuffle=False)

        history = []
        for epoch in range(epochs):
            train(model, train_loader, optimizer, criterion)
            val_loss, _ = evaluate(model, val_loader, criterion)
            history.append(val_loss)
            print(f'Strategy: {strategy}, Epoch: {epoch+1}, Val Loss: {val_loss}')

        results[strategy] = history

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for strategy, history in results.items():
        plt.plot(history, label=strategy)

    plt.title('Validation Loss vs. Epochs for Different Batch Sizing Strategies')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('progressive_batch_sizing_experiment/progressive_batch_sizing_comparison.png')
    plt.show()

if __name__ == '__main__':
    main()

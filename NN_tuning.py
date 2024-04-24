
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import optuna
import pickle

# Load and prepare data
df = pd.read_csv('preprocessed_gene_expression.csv')
x = df.drop(['death_from_cancer_Died of Disease', 'death_from_cancer_Living'], axis=1).values
y = df['death_from_cancer_Died of Disease'].values

# Neural network class
class NeuralNet(nn.Module):
    def __init__(self, num_features, num_layers, num_units):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(num_features if i == 0 else num_units, num_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(num_units, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

# Optimser
def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_units = trial.suggest_int('num_units', 50, 500)
    num_epochs = trial.suggest_int('epochs', 1, 20)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNet(X_train.shape[1], num_layers, num_units)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Validation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        outputs = model(X_test_tensor)
        loss = criterion(outputs, y_test_tensor)
    return loss.item()

# Results study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

# Save the best hyperparameters
with open('best_hyperparameters.pkl', 'wb') as f:
    pickle.dump(study.best_trial.params, f)

print("Best Hyperparameters:", study.best_trial.params)

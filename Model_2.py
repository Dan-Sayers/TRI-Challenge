import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Define neural network
class NeuralNet(torch.nn.Module):
    def __init__(self, in_features, first_layer_neurons, dropout_rate):
        super(NeuralNet, self).__init__()
        self.layer1 = torch.nn.Linear(in_features, first_layer_neurons)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.output = torch.nn.Linear(first_layer_neurons, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.output(x))
        return x
    
# Load model
checkpoint = torch.load('model_2.pth')
hyperparameters = checkpoint['hyperparameters']

# Load dataset
df = pd.read_csv('sample_gene_expression.csv')
true = np.array(df['death_from_cancer_Died of Disease'])
x_axis = df.drop(['death_from_cancer_Died of Disease','death_from_cancer_Living'], axis=1).values

# Instantiate model
model = NeuralNet(
    in_features=hyperparameters['in_features'],
    first_layer_neurons=hyperparameters['first_layer_neurons'],
    dropout_rate=hyperparameters['dropout_rate']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare DataLoader
X_tensor = torch.tensor(x_axis, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Make predictions
predictions = []
with torch.no_grad():
    for data in loader:
        outputs = model(data[0])
        predicted = (outputs.data > 0.5).float()
        predictions.extend(predicted.numpy().flatten())

# Print predictions
for predicted, actual in zip(predictions, true):
    print(f"Predicted Value: {int(predicted)} | Actual Value: {actual}")

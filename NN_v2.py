import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set Hyperparameters
BATCH_SIZE = 64  
LEARNING_RATE = 0.0002 
NUM_EPOCHS = 10
DROPOUT_RATE = 0.5
FIRST_LAYER_NEURONS = 175

# Load and prepare data
df = pd.read_csv('preprocessed_gene_expression.csv')
x_axis = df.drop(['death_from_cancer_Died of Disease', 'death_from_cancer_Living'], axis=1).values
y_axis = df['death_from_cancer_Died of Disease'].values
X_train, X_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

# Convert arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define Model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], FIRST_LAYER_NEURONS)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.output = nn.Linear(FIRST_LAYER_NEURONS, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.output(x))
        return x

# Initialize Model
model = NeuralNet()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# Training and validation
def train_model():
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# Validation
def validate_model():
    model.eval()
    y_pred = []
    with torch.no_grad():
        for data, _ in test_loader:
            outputs = model(data)
            predicted = (outputs.data > 0.5).float()
            y_pred.extend(predicted.numpy().flatten())
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Run training and validation
train_model()
validate_model()

# Saving the model
hyperparameters = {
    'in_features': model.layer1.in_features,
    'batch_size':BATCH_SIZE,
    'learning_rate':LEARNING_RATE,
    'num_epochs': NUM_EPOCHS,
    'dropout_rate': DROPOUT_RATE,
    'first_layer_neurons': FIRST_LAYER_NEURONS
}
torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters': hyperparameters
}, 'model_2.pth')

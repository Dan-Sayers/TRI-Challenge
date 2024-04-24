import pandas as pd
import numpy as np
import pickle

# Load the model
filename = 'model_1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Load the dataset
df = pd.read_csv('sample_gene_expression.csv')
true = np.array(df['death_from_cancer_Died of Disease'])

# Load preprocessing
scaler = pickle.load(open('scaler.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))

# Preprocess the data
x_axis = df.drop(['death_from_cancer_Died of Disease','death_from_cancer_Living'], axis=1) 
X_scaled = scaler.transform(x_axis)
X_pca = pca.transform(X_scaled)

# Predict using the loaded model
predictions = loaded_model.predict(X_pca)

# Print predictions
for predicted, actual in zip(predictions, true):
    print(f"Predicted Value: {int(predicted)} | Actual Value: {actual}")

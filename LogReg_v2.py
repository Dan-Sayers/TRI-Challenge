# Imports 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load data
df = pd.read_csv('preprocessed_gene_expression.csv')

# Perform PCA
scaler = MinMaxScaler()
x_axis = df.drop(['death_from_cancer_Died of Disease','death_from_cancer_Living'], axis=1)
y_axis = df['death_from_cancer_Died of Disease']
X_scaled = scaler.fit_transform(x_axis)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Define logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_axis, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Test and Validate Model
y_pred = model.predict(X_test)

# Plot outcomes and outputs
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the model
pickle.dump(model, open('model_1.sav', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(pca, open('pca.pkl', 'wb'))
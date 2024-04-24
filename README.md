# TRI engineer pre-interview test: Predictive modelling with the METABRIC dataset

This repository contains a collection of files used for processing data and developing machine learning models for making predictions on mortality. 
Below is an overview of each file and its purpose.


## PYTHON SCRIPTS OVERVIEW

These scripts preprocess the 'gene_expression' dataset and the use it to train 2 models. The first of which verifies the second. 
The second of which may be used for the prognosis of mortality of a patient given the variables available.

### Preprocessing.py
- **Purpose**: Handles data preprocessing including outlier detection, missing value handling, data normalization and encoding.
- **Features**: Detect and handle outliers, replace missing with average, remove missing or irrelevant values, min max normalisation, OHE

### LogReg_v2.py
- **Purpose**: Implements logistic regression model including PCA, training, testing, and saving the model.
- **Features**: Data loading, PCA, model definition, training, testing, outcome plotting, model saving ('model_1.sav').

### Model_1.py
- **Purpose**: Loads and utilises the pre-trained logistic regression ('model_1.sav') model to make predictions on provided datasets.
- **Features**: Model loading, data preprocessing, prediction making.

### NN_v2.py
- **Purpose**: Manages a complete workflow from setting hyperparameters, training, and validating a neural network, to saving the trained model.
- **Features**: Data preparation, model training/validation, model saving ('model_2.pth').

### Model_2.py
- **Purpose**: Loads and utilises the pre-trained neural network model ('model_2.pth') to make predictions on provided datasets.
- **Features**: Model definition, data loading, prediction execution.


#### INPUTS

These are the minimum input files required. This file may be used with Preprocessing.py to produce a dataset that can be used for training.

- **column_actions.csv**
  - **Description**: Contains specifications and actions to be applied to various columns during the preprocessing phase.
  - **Usage**: Read by `Preprocessing.py` to determine how to process each column of the input data.

- **gene_expression.csv**
  - **Description**: Raw gene expression dataset used as input for preprocessing and model training.
  - **Usage**: Loaded in scripts for initial data handling and transformations.


#### OUTPUTS

These are potential outputs.

- **preprocessed_gene_expression.csv**
  - **Description**: Resulting dataset after running Preprocessing.py and includes cleaning, normalization, and outlier removal.
  - **Usage**: Used as the cleaned dataset for training machine learning models.

- **scaler.pkl**
  - **Description**: Pickle file containing the scaler object used to normalize features in the dataset from running LogReg_v2.py.
  - **Usage**: Loaded during the model prediction phase to apply the same scaling transformation to new data.

- **pca.pkl**
  - **Description**: Pickle file containing the PCA model used to reduce the dimensionality of the dataset from running LogReg_v2.py.
  - **Usage**: Utilized in the model training and validation process to transform the data.

- **model_1.sav**
  - **Description**: Saved logistic regression model trained on the preprocessed dataset from running LogReg_v2.py.
  - **Usage**: Loaded for making predictions in a production environment.

- **model_2.pth**
  - **Description**: PyTorch model file containing a trained neural network from running NN_v2.py.
  - **Usage**: Used for inference to predict outcomes based on new gene expression data.


## SUPPLMENTRY FILES

These may be used to speed up the process of developing these models with the same method.

### NN_tuning.py
- **Purpose**: Hyperparameter tuning for neural networks using optimization techniques.
- **Features**: Network class definition, training loop, hyperparameter optimization.

### gene_expression_formatter.py
- **Purpose**: Formats gene expression datasets into structured DataFrames for analysis.
- **Features**: Load dataset and reformat for further processing.

### sample_gene_expression.csv
- **Purpose**: Sample input file formatted to match the required input structure for model predictions.
- **Usage**: Can be used as a template for preparing new gene expression data for predictions.


## INSTALLATION

To run these scripts, you need Python 3.x and the following packages:

```bash
pip install numpy pandas scikit-learn matplotlib pytorch
```

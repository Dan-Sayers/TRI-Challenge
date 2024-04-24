# Imports 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Define functions
def detect_and_handle_outliers(dataframe, column, whisker_width=1.5):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - whisker_width * IQR
    upper_bound = Q3 + whisker_width * IQR

    outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
    print(f"Number of outliers in {column}: {outliers.shape[0]}")

    sns.boxplot(x=dataframe[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

    dataframe = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

    return dataframe

def replace_missing_with_average(column: str, df):
    if column in df.columns:
        tumor_size_mean = int(df[column].mean())
        df[column] = df[column].fillna(tumor_size_mean)
        print(f"Filled missing {column} with mean value: {tumor_size_mean}")

# Load data
df = pd.read_csv('gene_expression.csv')

total_rows = len(df)
print(f"Orignal Number of Vals: {total_rows}")

# Load column actions
column_actions = pd.read_csv('column_actions.csv')
remove_columns = column_actions[column_actions['Action'] == 'remove']['Column Name'].tolist()
ohe_columns = column_actions[column_actions['Action'] == 'OHE']['Column Name'].tolist()
scale_columns = column_actions[column_actions['Action'] == 'scale']['Column Name'].tolist()

# Remove irrelevant vals
df_cleaned = df[df['death_from_cancer'] != 'Died of Other Causes']
df_cleaned = df_cleaned.drop(columns=remove_columns)

df_cleaned.reset_index(drop=True, inplace=True)

reduced_rows = len(df_cleaned)
print(f"Size after irrelevant vals: {reduced_rows}")

# Process or remove samples with missing vals
df_cleaned['mutation_count'] = df_cleaned['mutation_count'].fillna(0)
df_cleaned['3-gene_classifier_subtype'] = df_cleaned['3-gene_classifier_subtype'].replace('', 'nan').fillna('nan')

replace_missing_with_average('tumor_size', df_cleaned)

columns_to_check = []
missing_values = df_cleaned.isnull().sum()
#for col, val in missing_values.items():
#    print(f'{col}: {val}')

for column_name, missing_count in missing_values.items():
    if missing_count != 0:
        columns_to_check.append(column_name)

exceptions = [] # Fill with exceptional fields / 'tumor_stage'

columns_to_check = [field for field in columns_to_check if field not in exceptions]
df_cleaned = df_cleaned.dropna(subset=columns_to_check)

reduced_rows = len(df_cleaned)
print(f"Size after removeing  missing vals: {reduced_rows}")

#Idenitfy and remove outliers
df_cleaned = detect_and_handle_outliers(df_cleaned, 'age_at_diagnosis')

reduced_rows = len(df_cleaned)
print(f"Size after removeing  outliers: {reduced_rows}")

removed_percentage = ((total_rows-reduced_rows) / total_rows) * 100
print(f"Removed {total_rows-reduced_rows} samples from {total_rows} ({removed_percentage:.2f}%)")

df_cleaned.to_csv('cleaned_gene_expression.csv', index=False)

# Identify correlations
sns.histplot(data=df_cleaned, x='age_at_diagnosis', hue='death_from_cancer', multiple='stack')
plt.show()

sns.countplot(data=df_cleaned, x='pam50_+_claudin-low_subtype', hue='death_from_cancer')
plt.show()

#One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
encoded_data = pd.DataFrame()

for col in ohe_columns:
    encoded_features = encoder.fit_transform(df_cleaned[[col]])
    encoded_feature_names = encoder.get_feature_names_out([col])
    encoded_feature_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    encoded_data = pd.concat([encoded_data, encoded_feature_df], axis=1)

df_cleaned.reset_index(drop=True, inplace=True)
encoded_data.reset_index(drop=True, inplace=True)
df_formatted = pd.concat([df_cleaned, encoded_data], axis=1)
df_formatted = df_formatted.drop(columns=ohe_columns)

# Normalization
scaler = MinMaxScaler()
for col in scale_columns:
    df_formatted[col] = scaler.fit_transform(df_formatted[[col]])

# Save the cleaned and formatted data
df_formatted.to_csv('preprocessed_gene_expression.csv', index=False)
os.remove('cleaned_gene_expression.csv')
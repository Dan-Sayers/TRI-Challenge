import pandas as pd

# Load dataset
df = pd.read_csv('gene_expression.csv')

# Create a new DataFrame for actions
columns = df.columns
actions = [''] * len(columns)
column_actions = pd.DataFrame(list(zip(columns, actions)), columns=['Column Name', 'Action'])

column_actions.to_csv('column_actions.csv', index=False)

print("CSV file with column names has been created for manual actions input.")

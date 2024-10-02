import pandas as pd

# Load the CSV file to inspect the data and check which features are available
file_path = 'HR_Analytics.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()


import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'Attrition' column to numerical for correlation calculations: Yes = 1, No = 0
data['Attrition_Numeric'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Select only numeric columns for the correlation matrix
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Add 'Attrition_Numeric' to the numeric data
numeric_data['Attrition_Numeric'] = data['Attrition_Numeric']

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Extract the correlation of all features with 'Attrition_Numeric'
attrition_corr = correlation_matrix['Attrition_Numeric'].sort_values(ascending=False)


# Increase the number of features displayed on the plot (let's show the top 20 features)
plt.figure(figsize=(10, 10))  # Increase the figure size to accommodate more features
sns.barplot(x=attrition_corr.head(20), y=attrition_corr.head(20).index)
plt.title("Top 20 Features Correlated with Attrition (Absolute Values)")
plt.xlabel("Absolute Correlation with Attrition")
plt.ylabel("Features")
plt.tight_layout()

plt.show()



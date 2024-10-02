import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = 'HR_Analytics.csv'
data = pd.read_csv(file_path)

# Prepare data for plotting
attrition_by_gender = data.groupby('TotalWorkingYears')['Attrition'].value_counts(normalize=True).unstack() # Normalize om de proportie te krijgen, en verander gender om op andere zaken te groeperen

print(attrition_by_gender)

# Plotting
plt.figure(figsize=(10, 6))
attrition_by_gender.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])

plt.title('Attrition Rates by TotalWorkingYears')
plt.ylabel('Proportion')
plt.xlabel('TotalWorkingYears')
plt.xticks(rotation=90)
plt.legend(title='Attrition', labels=['YES', 'NO'])
plt.tight_layout()
plt.show()

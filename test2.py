import os
import joblib
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Dataset laden
data = pd.read_csv('HR_Analytics.csv')
# Kenmerken en target kiezen
# Encodering van 'Gender' (1 voor 'Male', 0 voor 'Female')
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
# One-hot encoding voor categorische variabelen zoals 'Department' en 'BusinessTravel'
data = pd.get_dummies(data, columns=['Department', 'BusinessTravel'], drop_first=True)
# Features instellen
features = ['Age', 'Gender', 'Education', 'TotalWorkingYears', 'MonthlyIncome'] + \
[col for col in data.columns if 'Department' in col or 'BusinessTravel' in col]
X = data[features]
# Target instellen (verondersteld dat we 'Attrition' willen voorspellen; aanpassen indien nodig)
y = data['Attrition'].map({'Yes': 1, 'No': 0}) # Attrition encoderen naar 1 voor 'Yes', 0 voor 'No'
# Train-test splitsen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Random Forest Model trainen
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Voorspellingen maken en rapport printen
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# Model opslaan
joblib.dump(model, 'hiring_model.pkl')


# Extract feature importances  
importances = model.feature_importances_  
  
# Create a DataFrame to hold feature names and their importance scores  
feature_importance_df = pd.DataFrame({  
    'Feature': features,  
    'Importance': importances  
})  
  
# Sort the DataFrame by importance scores in descending order  
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)  
  
# Display the feature importances  
print(feature_importance_df)


import lime
import lime.lime_tabular
import numpy as np
# Instantieer LIME explainability tool
explainer = lime.lime_tabular.LimeTabularExplainer(
training_data=np.array(X_train),
feature_names=features,
class_names=['No Attrition', 'Attrition'],
mode='classification'
)
# Kies een willekeurige sample uit de testset
i = 25 # Je kunt een andere index kiezen voor een ander individu
exp = explainer.explain_instance(
data_row=X_test.iloc[i],
predict_fn=model.predict_proba # Functie om voorspellingen te genereren
)
# Toon de uitleg in een Jupyter Notebook of exporteer naar een HTML bestand
exp.show_in_notebook(show_all=False)
# Indien je buiten een notebook werkt, kun je de uitleg ook opslaan als HTML
exp.save_to_file('lime_explanation.html')

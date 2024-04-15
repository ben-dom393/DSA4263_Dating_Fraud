import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = "../../data/processed/combined_profiles_sorted.xlsx"
data = pd.read_excel(file_path)

# Separate features and target variable
X = data.drop(columns=['scam'])
y = data['scam']

# Handle categorical variables using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Fit LassoCV to select features
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)

# Select features using LassoCV
sfm = SelectFromModel(lasso_cv)
X_train_selected = sfm.fit_transform(X_train, y_train)
X_test_selected = sfm.transform(X_test)

# Fit LassoLarsCV on selected features for debiasing
lassolars_cv = LassoLarsCV(cv=5)
lassolars_cv.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = lassolars_cv.predict(X_test_selected)

# Convert predicted probabilities to binary predictions
y_pred_binary = np.round(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)

# For multiclass classification, use appropriate average parameter setting
precision = precision_score(y_test, y_pred_binary, average='weighted')
recall = recall_score(y_test, y_pred_binary, average='weighted')
f1 = f1_score(y_test, y_pred_binary, average='weighted')

# Plot coefficients
plt.figure(figsize=(10, 6))
plt.plot(range(len(lassolars_cv.coef_)), lassolars_cv.coef_, marker='o', linestyle='none')
plt.title('Debiased Lasso Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.grid(True)
plt.show()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

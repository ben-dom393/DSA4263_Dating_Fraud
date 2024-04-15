import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = "C:/Users/xembr/DSA4263_Dating_Fraud/data/processed/combined_profiles_sorted.xlsx"
data = pd.read_excel(file_path)

# Separate features and target variable
X = data.drop(columns=['scam'])
y = data['scam']

# Handle categorical variables using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create the object of the Lasso model with regularization parameter alpha
lasso = Lasso(alpha=0.1)

# Fit the Lasso model
lasso.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso.predict(X_test)

# Convert predicted probabilities to binary predictions
y_pred_binary = np.round(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

# Plot coefficients
plt.figure(figsize=(10, 6))
plt.plot(range(len(lasso.coef_)), lasso.coef_, marker='o', linestyle='none')
plt.title('Lasso Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.grid(True)
plt.show()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

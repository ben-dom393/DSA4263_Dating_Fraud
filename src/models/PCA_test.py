import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# Load the train and test datasets
train_df = pd.read_csv("../../data/processed/train_profiles.csv")
test_df = pd.read_csv("../../data/processed/test_profiles.csv")

# Concatenate the train and test datasets
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Encode categorical variables (if necessary)
combined_df = pd.get_dummies(combined_df, columns=['location', 'ethnicity', 'occupation', 'status', 'country'])

# Standardize the numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(combined_df.drop(['scam', 'age_group', 'description'], axis=1))

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Visualize the first three principal components
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=combined_df['scam'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(3), pca.explained_variance_ratio_, marker='o', label='Component-wise explained variance ratio')
plt.plot(range(3), np.cumsum(pca.explained_variance_ratio_), marker='o', label='Cumulative explained variance ratio')
plt.title("Component-wise and Cumulative Explained Variance")
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.legend()
plt.show()
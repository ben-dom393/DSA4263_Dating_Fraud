import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from joblib import load

df = pd.read_csv("../../data/processed/test_profiles.csv")

y_test = df['scam']
X_test = df.drop(columns=['scam','age','location'])

# load models
encoder = load("../saved_models/ohe_encoder.joblib")
bow_vectorizer = load("../saved_models/bow_vectorizer.joblib")
LR = load("../saved_models/logistic_regression_trained.joblib")

# transform data
X_test_ohe = encoder.transform(X_test.drop(["description"], axis=1))
X_test_bow = bow_vectorizer.transform(X_test["description"])

# combine features
X_test_combined = np.hstack([X_test_ohe.toarray(), X_test_bow.toarray()])

# predict
y_pred = LR.predict(X_test_combined)
y_pred_proba = LR.predict_proba(X_test_combined)[:,1]

# calculate metrics
results = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred_proba)]

# print results
print("Accuracy: {:.2f}".format(results[0]))
print("Precision: {:.2f}".format(results[1]))
print("Recall: {:.2f}".format(results[2]))
print("F1: {:.2f}".format(results[3]))
print("ROC-AUC: {:.2f}".format(results[4]))

# print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

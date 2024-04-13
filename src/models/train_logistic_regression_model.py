import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

from joblib import dump

# read data
df = pd.read_csv("../../data/processed/train_profiles.csv")
y = df['scam']
X = df.drop(columns=['scam','age','location'])

# get significant unigrams using BoW and LR feature weights
bow_vectorizer = CountVectorizer(stop_words=stopwords.words("english"),
                                 ngram_range=(1,3))
X_bow = bow_vectorizer.fit_transform(X["description"])

LR_bow = LogisticRegression(max_iter=1000)
LR_bow.fit(X_bow, y)

feature_importance = pd.DataFrame(zip(bow_vectorizer.get_feature_names_out(),
                                      np.transpose(LR_bow.coef_).flatten()),
                                      columns=['features', 'coef']) 

# get outliers using Tukey's method
stats = feature_importance["coef"].describe()
iqr = stats["75%"] - stats["25%"]
k = 15 # 15 is the optimal value found from tuning in notebooks/profiles_logistic_regression.ipynb
outliers = feature_importance[(feature_importance["coef"] < stats["25%"] - iqr * k) | (feature_importance["coef"] > stats["75%"] + iqr * k)]
vocab = outliers

# train final model with categorical features and most discriminative vocab
# categorical features
encoder = OneHotEncoder(handle_unknown="ignore")
X_train_ohe = encoder.fit_transform(X.drop(["description"], axis=1))

# BoW features
bow_vectorizer = CountVectorizer(stop_words=stopwords.words("english"),
                                 vocabulary=vocab)
X_train_bow = bow_vectorizer.fit_transform(X["description"])

X_train_combined = np.hstack([X_train_ohe.toarray(), X_train_bow.toarray()])

LR_combined = LogisticRegression(max_iter=1000)
LR_combined.fit(X_train_combined, y)

# save all the models
dump(encoder, '../../models/ohe_encoder.joblib')
dump(bow_vectorizer, '../../models/bow_vectorizer.joblib')
dump(LR_combined, '../../models/logistic_regression_trained.joblib') 
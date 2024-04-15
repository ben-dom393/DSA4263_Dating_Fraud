import os

import pandas as pd
import numpy as np

from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import argparse

# Function to load and preprocess images
def load_images(image_paths):
    images = []
    for image_path in image_paths:
        img = load_img(image_path, target_size=(128, 128), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform prediction on a dataset with image and profile features')

    parser.add_argument('data_path', type=str, help='Path to the dataset to perform prediction on')
    parser.add_argument('--output_path', type=str, default='.', help='Path to save the results')

    args = parser.parse_args()

    # Load combined dataset
    df = pd.read_csv(args.data_path)

    # Separate features for both models
    profile_features = df.drop(columns=['scam', 'image_path', 'face_fake', 'age', 'location'])
    face_image_paths = df['image_path']

    # Load models
    encoder = load("../models/ohe_encoder.joblib")
    bow_vectorizer = load("../models/bow_vectorizer.joblib")
    LR = load("../models/logistic_regression_trained.joblib")
    face_model = load_model("../models/base_model_best.h5")
    face_model.load_weights("../models/base_model_best.weights.h5")

    # Predict using profile model
    X_test_ohe = encoder.transform(profile_features.drop(['description'], axis=1))
    X_test_bow = bow_vectorizer.transform(profile_features['description'])
    X_test_combined = np.hstack([X_test_ohe.toarray(), X_test_bow.toarray()])
    profile_pred_prob = LR.predict_proba(X_test_combined)[:, 1]

    # Predict using face model
    X_test_images = load_images(face_image_paths)
    face_pred_prob = face_model.predict(X_test_images).flatten()

    # Combine predictions
    combined_prob = (profile_pred_prob + face_pred_prob) / 2

    # Weighted combination if one model is more reliable
    # For example, if the profile model is more reliable:
    weighted_combined_prob = 0.4 * profile_pred_prob + 0.6 * face_pred_prob

    # As both models are performing well, an OR gate can be set up to give a binary classification the predicted labels
    predicted_labels = np.where((profile_pred_prob > 0.5) | (face_pred_prob > 0.5), 1, 0)

    # Output combined results
    results_df = pd.DataFrame({
        'true_profile_scam': df['scam'],
        'profile_prob': profile_pred_prob,
        'true_face_fake': df['face_fake'],
        'face_prob': face_pred_prob,
        'combined_prob': combined_prob,
        'weighted_prob': weighted_combined_prob,
        'predicted_label': predicted_labels

    })
    results_df.to_csv(os.path.join(args.output_path, "results.csv"), index=False)


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_image

# Ensure proper backend setup for matplotlib in non-interactive environments
plt.switch_backend('agg')

# Directories containing preprocessed images
save_directory_real = 'data/interim/face_processed/real'
save_directory_fake = 'data/interim/face_processed/fake'

# Function to create a DataFrame with image paths and labels
# Changing labelling to be 1 if fake and 0 if real
def create_image_label_df(real_dir, fake_dir):
    # Check if directories exist and contain files
    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        raise ValueError("One or both directories do not exist or are not accessible.")

    # List and filter files in the directories
    real_images = [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir)
                   if os.path.isfile(os.path.join(real_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_images = [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir)
                   if os.path.isfile(os.path.join(fake_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Print the number of images found in each directory
    print(f"Folder: {real_dir}, Number of images: {len(real_images)}")
    print(f"Folder: {fake_dir}, Number of images: {len(fake_images)}")

    if not real_images and not fake_images:
        raise ValueError("Both directories are empty or do not contain valid image files.")

    # Combine real and fake images into one DataFrame
    df = pd.DataFrame(real_images + fake_images, columns=['image_path', 'label'])
    return df

# Create the DataFrame
images_df = create_image_label_df(save_directory_real, save_directory_fake)

# Splitting the data into features and target arrays
X = images_df['image_path'].values
y = images_df['label'].values

# First split to separate out the training set (80%) and the remaining data (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

# Second split to separate out the validation set (50% of the remaining data) and the test set (50% of the remaining data)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp, shuffle=True)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Testing set size: {len(X_test)}")

# Directory for saving test images
test_image_save_dir = 'saved_test_images'
os.makedirs(test_image_save_dir, exist_ok=True)

# Subdirectories for real and fake images
real_dir = os.path.join(test_image_save_dir, 'test_real')
fake_dir = os.path.join(test_image_save_dir, 'test_fake')
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

def save_test_images(image_paths, labels):
    for idx, (path, label) in enumerate(zip(image_paths, labels)):
        # Load the image
        img = load_img(path, target_size=(128, 128), color_mode='grayscale')
        
        # Determine the subdirectory based on the label
        sub_dir = real_dir if label == 0 else fake_dir
        
        # Prepare file name and save path
        filename = f'test_img_{idx}.png'
        save_path = os.path.join(sub_dir, filename)
        
        # Save image
        img.save(save_path)

# Example usage of the function
save_test_images(X_test, y_test)

def load_images(image_paths, labels):
    images = []
    for image_path in image_paths:
        img = load_img(image_path, target_size=(128, 128), color_mode='grayscale') #Note: gray scale and size pre-processed alr
        img_array = img_to_array(img)/255.0 #Convert image to array + normalise 
        images.append(img_array) 
    return np.array(images), np.array(labels)

# Load and get images in dataset (From path)
X_train_processed, y_train_processed = load_images(X_train, y_train)
X_val_processed, y_val_processed = load_images(X_val, y_val)
X_test_processed, y_test_processed = load_images(X_test, y_test)

# Model Construction
model = Sequential()
model.add(Input(shape=(128, 128, 1)))  # Explicit input layer
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu", kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks_list = [
    EarlyStopping(monitor='val_loss', 
                  patience=5,
                  restore_best_weights=True,verbose=0),
    ReduceLROnPlateau(monitor = 'val_loss',patience=2,
                      factor=0.5,
                      min_lr = 0.00001,
                      verbose = 1)
]

# Setup the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    'model_weights_best.weights.h5',       # Path where the model will be saved
    monitor='val_accuracy',        # Metric to monitor
    save_best_only=True,           # Save only the best model
    save_weights_only=True,        # Save only the weights
    mode='max',                    # Save the model when the monitored metric is maximized
    verbose=1                      # Log when saving models
)

# Add the checkpoint callback to the list of callbacks
callbacks_list.append(checkpoint_callback)

mod_hist = model.fit(X_train_processed, y_train_processed,
                     epochs=30, batch_size=32,
                     validation_data=(X_val_processed, y_val_processed),
                     callbacks=callbacks_list,
                     verbose=1)

# Save the whole model
model.save('base_model_best.h5')

# Alternatively, save just the weights
model.save_weights('base_model_best.weights.h5')

# After training the model, extract history and save to a CSV
history_df = pd.DataFrame(mod_hist.history)
history_df.to_csv('training_history.csv', index=False)

# plots for accuracy and Loss with epochs
error = pd.DataFrame(mod_hist.history)

plt.figure(figsize=(18,5),dpi=200)
sns.set_style('darkgrid')

plt.subplot(121)
plt.title('Cross Entropy Loss',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.plot(error['loss'], label='Training Loss')
plt.plot(error['val_loss'], label='Validation Loss')
plt.legend()

plt.subplot(122)
plt.title('Classification Accuracy',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.plot(error['accuracy'], label='Training Accuracy')
plt.plot(error['val_accuracy'], label='Validation Accuracy')
plt.legend()

plt.savefig('training_performance.png')

# Load the best weights before evaluating
model.load_weights('model_weights_best.weights.h5')

# Evaluate the model on the test set
# Print and save test accuracy and loss
test_metrics = model.evaluate(X_test_processed, y_test_processed, verbose=1)
test_accuracy = test_metrics[1]  # Assuming accuracy is the second metric you're tracking
print(f'Test Accuracy: {test_accuracy}')

# Save test metrics to a file
test_metrics_df = pd.DataFrame({
    'test_loss': [test_metrics[0]],
    'test_accuracy': [test_accuracy]
})
test_metrics_df.to_csv('test_metrics.csv', index=False)

# Generate predictions (probabilities -- the output of the last layer) on test set
#Classes: fake == 1, real==0
y_probab = model.predict(X_test_processed)

# Prepare DataFrame with test paths, true labels, and predicted probabilities
results_df = pd.DataFrame({
    'X_test': X_test,      # Image paths
    'y_test': y_test,      # True labels
    'y_probab': y_probab.flatten()  # Predicted probabilities
})

# Save the DataFrame to a CSV file
results_df.to_csv('model_predictions.csv', index=False)

## CONFUSION MATRIX

# Generate predictions
y_pred = (model.predict(X_test_processed) > 0.5).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()  # This prevents the plot from showing inline if you're running this in a Jupyter notebook or similar environment.

## AUC CURVE
# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probab)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()  # This also prevents the plot from showing inline if you're running this in a Jupyter notebook or similar environment.

########### Explanability ####################
#######LIME
def predict4lime(img2):
    # Uncomment below to check the input shape
    # print(img2.shape)
    # Ensure the input has the correct shape for the model
    return model.predict(img2[:, :, :, 0:1])  # Assuming the model expects a 4D input

# Initialize the explainer
explainer = lime_image.LimeImageExplainer()

# Generate predictions (probabilities -- the output of the last layer) on test set
y_probab = model.predict(X_test_processed)

# Convert probabilities to binary predictions
y_pred = (y_probab.flatten() > 0.5).astype(int)

# Add binary predictions to results DataFrame
results_df['y_pred'] = y_pred

# Function to select images based on specified criteria
# Function to select images based on specified criteria
def select_images_for_lime(df, category, n=10, threshold=0.45):
    if category == 'incorrect':
	# We assume that 'y_pred' is binary with 0.5 as threshold.
        df['incorrectness'] = np.abs(df['y_probab'] - 0.5)
        selection = df.nlargest(n, 'incorrectness')
    elif category == 'edge_cases':
        # Closest to 0.5 are the edge cases.
        df['closeness_to_threshold'] = np.abs(df['y_probab'] - 0.5)
        selection = df.nsmallest(n, 'closeness_to_threshold')
    elif category == 'high_confidence':
	# We take the probabilities closest to 0 or 1 for high confidence.
        df['confidence'] = np.maximum(df['y_probab'], 1 - df['y_probab'])
        selection = df.nlargest(n, 'confidence')
    else:
        raise ValueError("Unknown category for selection.")

    # Drop the temporary columns we created for selection
    df.drop(columns=['incorrectness', 'closeness_to_threshold', 'confidence'], errors='ignore', inplace=True)

    return selection['X_test'].values, selection['y_test'].values, selection['y_probab'].values

# Function to visualize and save LIME explanation images
def save_lime_explanations(image_paths, labels, probabilities, subset_name, save_directory):
    for idx, (image_path, label, probability) in enumerate(zip(image_paths, labels, probabilities)):
        img = load_img(image_path, target_size=(128, 128), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape(1, *img_array.shape)

        explanation = explainer.explain_instance(img_array.squeeze(), predict4lime, top_labels=1, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

        # Process image and mask for visualization
        temp_rgb = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        red_color_adjust = np.full(temp_rgb.shape, 0.1)
        temp_rgb = np.where(mask_rgb, temp_rgb * (1 - red_color_adjust) + red_color_adjust, temp_rgb)

        # Create a plot
        fig, ax = plt.subplots()
        ax.imshow(temp_rgb)
        ax.axis('off')
        
        # Title with predicted label for clarity
        title = f'Predicted: {"Fake" if label == 1 else "Real"}, Prob: {probability:.5f}'
        ax.set_title(title)

        # Save the figure to a file
        file_name = f'{subset_name}_explanation_{idx}.png'
        file_path = os.path.join(save_directory, file_name)
        fig.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved: {file_path}")

# Ensure the save directory exists
save_directory = 'lime_explanations'
os.makedirs(save_directory, exist_ok=True)

incorrect_paths, incorrect_labels, incorrect_prob = select_images_for_lime(results_df, 'incorrect')
save_lime_explanations(incorrect_paths, incorrect_labels, incorrect_prob, 'incorrect', save_directory)

edge_cases_paths, edge_cases_labels, edge_cases_prob = select_images_for_lime(results_df, 'edge_cases')
save_lime_explanations(edge_cases_paths, edge_cases_labels, edge_cases_prob, 'edge_cases', save_directory)

high_confidence_paths, high_confidence_labels, high_confidence_prob = select_images_for_lime(results_df, 'high_confidence')
save_lime_explanations(high_confidence_paths, high_confidence_labels, high_confidence_prob, 'high_confidence', save_directory)

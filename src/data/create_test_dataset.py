######################################################################################
### Create test dataset by combining real and fake profiles with real and AI faces ###
######################################################################################

import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

random.seed(42)

test_profiles = pd.read_csv("../../data/processed/test_profiles.csv")
scam_profiles_ratio = 0.1

# Profiles #
# ------------------------------ #
real_profiles_idx = np.where(test_profiles['scam'] == 0)[0]
fake_profiles_idx = np.where(test_profiles['scam'] == 1)[0]

num_real_profiles = len(real_profiles_idx)
num_to_extract = int(num_real_profiles * (scam_profiles_ratio / 3))

real_profiles_for_real_images_idx, real_profiles_for_fake_images_idx =\
    train_test_split(real_profiles_idx, test_size=num_to_extract, random_state=42)

real_profiles_for_real_images, real_profiles_for_fake_images =\
    test_profiles.iloc[real_profiles_for_real_images_idx], test_profiles.iloc[real_profiles_for_fake_images_idx]

fake_profiles = random.choices(fake_profiles_idx, k=num_to_extract*2)
fake_profiles_for_real_images_idx, fake_profiles_for_fake_images_idx =\
    fake_profiles[:num_to_extract], fake_profiles[num_to_extract:]
# ------------------------------ #


# Images #
# ------------------------------ #
# Directories containing images
real_face_directory = '../../data/interim/face_processed/real'
fake_face_directory = '../../data/interim/face_processed/fake'

# list all image files in the directories
real_images_list = [f for f in os.listdir(real_face_directory) if os.path.isfile(os.path.join(real_face_directory, f))]
fake_images_list = [f for f in os.listdir(fake_face_directory) if os.path.isfile(os.path.join(fake_face_directory, f))]

# sample 
real_images_paths = [(os.path.join(real_face_directory, f), 0) for f in random.choices(real_images_list, k=num_real_profiles)]
fake_images_paths = [(os.path.join(fake_face_directory, f), 1) for f in random.choices(fake_images_list, k=num_to_extract*2)]

real_image_for_real_profiles, real_image_for_fake_profiles =\
    train_test_split(real_images_paths, test_size=num_to_extract, random_state=42)

fake_image_for_real_profiles, fake_image_for_fake_profiles =\
    train_test_split(fake_images_paths, test_size=num_to_extract, random_state=42)
# ------------------------------ #


# combine into 4 categories
# ------------------------------ #
# add column "image_path" to the profiles
test_profiles.insert(0, 'image_path', None)

print("real_profiles_real_images size: ", len(real_profiles_for_real_images_idx))
print("real_image_for_real_profiles size: ", len(real_image_for_real_profiles))

# real profiles with real images
test_profiles.loc[real_profiles_for_real_images_idx, 'image_path'] =\
    real_image_for_real_profiles

# real profiles with fake images
test_profiles.loc[real_profiles_for_fake_images_idx, 'image_path'] =\
    fake_image_for_real_profiles

# fake profiles with real images
test_profiles.loc[fake_profiles_for_real_images_idx, 'image_path'] =\
    real_image_for_fake_profiles

# fake profiles with fake images
test_profiles.loc[fake_profiles_for_fake_images_idx, 'image_path'] =\
    fake_image_for_fake_profiles


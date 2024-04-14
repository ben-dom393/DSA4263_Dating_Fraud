import json
import os
import pandas as pd

import sys

def create_dataframe(profiles):
    data = []
    for profile in profiles:
        with open(profile, "r", encoding="utf-8") as f:
            scam_profile = json.load(f)

        data.append(scam_profile)

    return pd.DataFrame(data)

def merge(output_path, *folder_paths):
    df_list = []
    for folder_path in folder_paths:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
        df = create_dataframe(files)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    # run script
    # python combine_dataset.py [output_path] [folder_path_1] [folder_path_2] [folder_path_3] ...

    # example
    # python combine_dataset.py ../../data/real_profiles.csv ../../data/real_profiles_1 ../../data/real_profiles_2 ../../data/real_profiles_3

    output_path = sys.argv[1]
    folder_paths = sys.argv[2:]

    # input folder paths with data to be merged
    merge(output_path, *folder_paths)

import json
import os
import pandas as pd

scam_profile_path_1 = os.path.join("..", "..", "data")
scam_profile_path_2 = os.path.join("..", "..", "data")

# read all json files from scam_profile_path_1
scam_profile_files_1 = [os.path.join(scam_profile_path_1, f) for f in os.listdir(scam_profile_path_1)]


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
    output_path = os.path.join("..", "..", "data", "merged_dataset.csv")
    scam_profile_path_2012_2015 = os.path.join("..", "..", "data", "scam_profiles_2012_2015")
    scam_profile_path_2016_2020 = os.path.join("..", "..", "data", "scam_profiles_2016_2020")

    # input folder paths with data to be merged
    merge(output_path, scam_profile_path_2012_2015, scam_profile_path_2016_2020)

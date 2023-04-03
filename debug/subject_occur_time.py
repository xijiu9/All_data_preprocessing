import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm
import matplotlib.pyplot as plt
import IPython
import numpy as np

print(time.asctime(time.localtime()))

with open("../Extract_Relation/Triple_list_after_2020.json", "r") as f:
    Triple_list = json.load(f)

for triple in tqdm(Triple_list):
    data_short = triple["data_short"]

    first_relation = triple["extract relation pairs"][list(triple["extract relation pairs"].keys())[0]]
    Triple_feature = first_relation["sub"][0] + ' & ' + first_relation["obj"][0]
    print(Triple_feature)

    subject_path = f"/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Single_intersection/{data_short}/Subject_dict.json"

    with open(subject_path) as f:
        Subject_dict = json.load(f)

    first_relation = triple["extract relation pairs"][list(triple["extract relation pairs"].keys())[0]]

    Sub_time_dict = {}

    for key in Subject_dict[first_relation["sub"][0]]:
        key = int(key)

        if key > 10000000:
            try:
                with open(f"../Fetch_Date_Info/only_date_map/{5000 * (key // 5000)}-{5000 * (key // 5000 + 1)}.json") as f:
                    time_map = json.load(f)
            except:
                print("wrong")
                print(f"../Fetch_Date_Info/data_map/{5000 * (key // 5000)}-{5000 * (key // 5000 + 1)}.json")
                continue
        else:
            continue

        year, month, day = int(time_map[f"{key}"]["full"]["EDAT"][0:4]), int(time_map[f"{key}"]["full"]["EDAT"][5:7]), int(
            time_map[f"{key}"]["full"]["EDAT"][8:10])
        full_time = time_map[f"{key}"]["full"]["EDAT"]

        Sub_time_dict[key] = full_time

    os.makedirs(f"Sub_result/{data_short}", exist_ok=True)
    with open(f"Sub_result/{data_short}/test_Sub.json", "w") as f:
        new_triple_list = json.dump(Sub_time_dict, f, indent=4)
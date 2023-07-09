import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import IPython
import numpy as np

print(time.asctime(time.localtime()))

triples_with_relation_path = sorted(glob.glob("Triple with Relation/BioLAMA/*/*/New_Triple_list.json"))
all_time_map = {}
all_triples, all_after_2023_triples = 0, 0

for key in trange(0, 37000000, 5000):
    try:
        with open(
                f"../Fetch_Date_Info/data_map/{str(5000 * (key // 5000)).zfill(8)}-{str(5000 * (key // 5000 + 1)).zfill(8)}.json") as f:
            time_map = json.load(f)
            all_time_map[str(5000 * (key // 5000)).zfill(8)] = time_map
    except:
        print(f"can not load {str(5000 * (key // 5000)).zfill(8)}-{str(5000 * (key // 5000 + 1)).zfill(8)}.json")

for file_path in triples_with_relation_path:
    try:
        with open(file_path, "r") as f:
            new_triple_list = json.load(f)
    except:
        continue

    print(file_path)
    print(len(new_triple_list))
    all_triples += len(new_triple_list)

    for triple in tqdm(new_triple_list):
        Trikeys = triple["extract relation pairs"].keys()
        Trikeys = [int(num) for num in Trikeys]

        time_stamp_1 = "2022/01/01 00:00"
        time_stamp_2 = "2022/04/01 00:00"
        before_abstract, middle_abstract, after_abstract = 0, 0, 0

        for key in Trikeys:
            try:
                time_map = all_time_map[str(5000 * (key // 5000)).zfill(8)]
            except:
                import IPython
                IPython.embed()

            if "EDAT" not in time_map[f"{key}"]["full"]:
                print(key)
                print(time_map[f"{key}"]["full"])
                print(time_map[f"{key}"]["full"].keys())
                exit(0)
            triple["extract relation pairs"][f"{key}"]["time"] = time_map[f"{key}"]["full"]["EDAT"]

            if time_map[f"{key}"]["full"]["EDAT"] < time_stamp_1:
                before_abstract += 1
            elif time_map[f"{key}"]["full"]["EDAT"] < time_stamp_2:
                middle_abstract += 1
            else:
                after_abstract += 1

        print(before_abstract, middle_abstract, after_abstract)
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

triples_with_relation_path = sorted(glob.glob("Triple with Relation/*/*/*/New_Triple_list.json"))

print(time.asctime(time.localtime()))

Triple_list_after_2020 = []
len_list = []

all_triples, all_after_2023_triples = 0, 0
for file_path in triples_with_relation_path:
    if "P2176" not in file_path:
        continue
    try:
        with open(file_path, "r") as f:
            new_triple_list = json.load(f)
    except:
        continue

    print(file_path)
    print(len(new_triple_list))
    all_triples += len(new_triple_list)

    part_triples_after_2023 = 0

    for triple in tqdm(new_triple_list):
        if triple["subject"] != "COVID-19":
            continue

        print(triple["subject"])

        Trikeys = triple["extract relation pairs"].keys()
        Trikeys = [int(num) for num in Trikeys]

        min_key = np.min(np.array(Trikeys))
        if min_key > 25000000:
            min_year = 10000
            max_year = 0
            min_date = "2024/01/01 06:00"
            max_date = "2019/12/31 06:00"
            for key in tqdm(Trikeys):
                with open(f"../Fetch_Date_Info/data_map/{5000 * (key // 5000)}-{5000 * (key // 5000 + 1)}.json") as f:
                    time_map = json.load(f)

                if "EDAT" not in time_map[f"{key}"]["full"]:
                    print(key)
                    print(time_map[f"{key}"]["full"])
                    print(time_map[f"{key}"]["full"].keys())
                    exit(0)
                triple["extract relation pairs"][f"{key}"]["time"] = time_map[f"{key}"]["full"]["EDAT"]
                year, month, day = int(time_map[f"{key}"]["full"]["EDAT"][0:4]), int(time_map[f"{key}"]["full"]["EDAT"][5:7]), int(time_map[f"{key}"]["full"]["EDAT"][8:10])

                if year <= 2019:
                    min_year = year
                    import IPython
                    IPython.embed()
                    break

                min_year = min(year, min_year)
                max_year = max(year, max_year)
                min_date = min(time_map[f"{key}"]["full"]["EDAT"], min_date)
                max_date = max(time_map[f"{key}"]["full"]["EDAT"], max_date)

            import IPython
            IPython.embed()

            if min_year >= 2020:
                triple["extract relation pairs"]["min time"] = min_date
                triple["extract relation pairs"]["max time"] = max_date
                all_after_2023_triples += 1
                part_triples_after_2023 += 1

                Triple_list_after_2020.append(triple)
                len_list.append(len(Trikeys))

#     print(part_triples_after_2023)
#
#     with open("Triple_list_after_2020.json", "w") as f:
#         new_triple_list = json.dump(Triple_list_after_2020, f, indent=4)
#
# print(all_triples, all_after_2023_triples)
#
# with open("Triple_list_after_2020.json", "w") as f:
#     new_triple_list = json.dump(Triple_list_after_2020, f, indent=4)
#
# plt.hist(len_list, bins=range(min(len_list), max(len_list) + 2, 1), log=True)
# plt.xscale('log')
#
# plt.savefig("len_list_triple_after_2023.png")
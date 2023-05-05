import logging
import time
import argparse
import os
import json
from datetime import datetime, timedelta
from tqdm import tqdm, trange
from copy import deepcopy

with open("Triple_list_after_2020.json", "r") as f:
    Triple_list = json.load(f)

Triple_list_updated = []
for triple in Triple_list:
    if "_hard" in triple["data_path"] and "medlama" in triple["data_path"]:
        continue

    extract_pairs = triple["extract relation pairs"]

    pmid_dict = {}
    for pmid, subobj in extract_pairs.items():
        if "time" in pmid:
            continue

        for sub in list(set(subobj["sub"])):
            for obj in list(set(subobj["obj"])):
                feature = sub + "&&&" + obj
                if feature not in pmid_dict.keys():
                    pmid_dict[feature] = [[pmid, subobj["time"]]]
                else:
                    pmid_dict[feature].append([pmid, subobj["time"]])

    for triple_feature, pmid_list in pmid_dict.items():
        new_triple = deepcopy(triple)
        new_triple["extract relation pairs"] = {}
        sub, obj = triple_feature.split("&&&")

        min_date = "2024/01/01 06:00"
        max_date = "2019/12/31 06:00"
        for pmid_time in pmid_list:
            pmid, time = pmid_time[0], pmid_time[1]
            new_triple["extract relation pairs"][pmid] = {"sub": [sub], "obj": [obj], "time":time}

            min_date = min(time, min_date)
            max_date = max(time, max_date)

        new_triple["extract relation pairs"]["min time"] = min_date
        new_triple["extract relation pairs"]["max time"] = max_date
        Triple_list_updated.append(new_triple)

print(len(Triple_list))
print(len(Triple_list_updated))
with open("Triple_list_after_2020_updated.json", "w") as f:
    json.dump(Triple_list_updated, f, indent=4)


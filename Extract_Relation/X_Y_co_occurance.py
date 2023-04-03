import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm
import copy
import pyfiglet
import IPython


def flatten(lst):
    # print(11111)
    result = []
    for item in lst:
        if isinstance(item, str) and item.startswith('[') and item.endswith(']'):
            try:
                eval_lst = eval(item)
            except:
                # print(item)
                eval_lst = [item[1:-1]]
                # print(eval_lst)
            result.extend(eval_lst)
        elif isinstance(item, str):
            result.append(item)
        elif isinstance(item, list):
            result.extend(flatten(item))
    return result


with open("../Extract_Relation/Triple_list_after_2020.json", "r") as f:
    Triple_list = json.load(f)

for triple in tqdm(Triple_list):
    short_path = triple["data_short"]

    object_path = f"Words_PMID/Single_intersection/{short_path}/Object_dict.json"
    subject_path = f"Words_PMID/Single_intersection/{short_path}/Subject_dict.json"
    with open(object_path) as f:
        Object_dict = json.load(f)
    with open(subject_path) as f:
        Subject_dict = json.load(f)

    cooccurance_subject_list = []


    first_relation = triple["extract relation pairs"][list(triple["extract relation pairs"].keys())[0]]
    Triple_feature = first_relation["sub"][0] + ' & ' + first_relation["obj"][0]
    OO = first_relation["obj"][0]

    for sub, pmids in Subject_dict.items():
        pmid_set = set(pmids)

        if pmid_set.intersection(set(Object_dict[OO])):
            cooccurance_subject_list.append(sub)

    os.makedirs(f"Co_occurance/{short_path}/{Triple_feature}", exist_ok=True)
    with open(f"Co_occurance/{short_path}/{Triple_feature}/Co_occur.json", "w") as f:
        json.dump(cooccurance_subject_list, f, indent=4)


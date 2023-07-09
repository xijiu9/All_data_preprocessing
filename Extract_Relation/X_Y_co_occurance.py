import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm
import copy
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


Relation_path = sorted(list(set(glob.glob("Triple_result/BioLAMA/*/*/", recursive=True))))
ALL_Subject, ALL_Object = {}, {}

for file_path in Relation_path:
    short_path = "/".join(file_path.split("/")[-4:-1])
    print(short_path)

    object_path = f"Words_PMID/Exact_intersection/{short_path}/Object_dict.json"
    subject_path = f"Words_PMID/Exact_intersection/{short_path}/Subject_dict.json"
    with open(object_path) as f:
        Object_dict = json.load(f)
    with open(subject_path) as f:
        Subject_dict = json.load(f)

    print(len(Object_dict), len(Subject_dict))
    old, new, zero = 0, 0, 0
    for k in Object_dict:
        if k not in ALL_Object.keys():
            ALL_Object[k] = Object_dict[k]
            new += 1
        else:
            try:
                assert set(ALL_Object[k]) == set(Object_dict[k])
            except:
                import IPython
                IPython.embed()
            old += 1

        if len(Object_dict[k]) == 0:
            zero += 1
    print(old, new, zero)

    old, new, zero = 0, 0, 0
    for k in Subject_dict:
        if k not in ALL_Subject.keys():
            ALL_Subject[k] = Subject_dict[k]
            new += 1
        else:
            assert set(ALL_Subject[k]) == set(Subject_dict[k])
            old += 1

        if len(Subject_dict[k]) == 0:
            zero += 1
    print(old, new, zero)

    with open(f"Words_PMID/Exact_intersection/ALL_Object_dict.json", "w") as f:
        json.dump(ALL_Object, f)
    with open(f"Words_PMID/Exact_intersection/ALL_Subject_dict.json", "w") as f:
        json.dump(ALL_Subject, f)
# all_values = list(ALL_Object.values())
#
# if len(all_values) >= 2:
#     common_keys = set(all_values[0]).intersection(all_values[1])
#     if common_keys:
#         print("头两个值有公共的键（key）:", len(common_keys))
#     else:
#         print("头两个值没有公共的键（key）")
#
import IPython
IPython.embed()




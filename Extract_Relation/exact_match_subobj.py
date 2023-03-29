import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm
import copy
import pyfiglet


Relation_path = sorted(list(set(glob.glob("Extract Result/e_result/MedLAMA/*/*/", recursive=True))))

tokenizer = nltk.RegexpTokenizer(r'[\w-]+')

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

# print(Relation_path)
for file_path in Relation_path:
    short_path = "/".join(file_path.split("/")[-4:-1])
    print(short_path)
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "Start Loading")
    with open(f'{file_path}/Object_dict.json') as f:
        Object_dict = json.load(f)
    with open(f'{file_path}/Subject_dict.json') as f:
        Subject_dict = json.load(f)
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "Finish Loading 1")

    data_short = os.path.join("Triple_result", short_path)

    with open(f'{data_short}/Object_list.json') as f:
        Object_list = json.load(f)
    with open(f'{data_short}/Subject_list.json') as f:
        Subject_list = json.load(f)
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "Finish Loading 2")
    with open(f'{data_short}/Triple_list.json') as f:
        Triple_list = json.load(f)

    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "Finish Loading 3")

    common_obj_pmid_dict = {}
    Tokenized_obj_dict = {}
    for obj in tqdm(Object_list):
        obj = obj.lower()
        words = tokenizer.tokenize(obj)
        Tokenized_obj_dict[obj] = words

    for obj, words in tqdm(Tokenized_obj_dict.items()):
        common_obj_pmid = set(Object_dict.get(words[0], []))
        for word in words[1:]:
            common_obj_pmid.intersection_update(Object_dict.get(word, []))
        common_obj_pmid_dict[obj] = list(common_obj_pmid)


        # for idx, word in enumerate(words):
        #     if word not in Object_dict.keys():
        #         continue
        #     word_pmid = set(Object_dict[word])
        #     if idx == 0:
        #         common_obj_pmid = set(word_pmid)
        #     else:
        #         common_obj_pmid = common_obj_pmid.intersection(word_pmid)
        # common_obj_pmid_dict[obj] = list(common_obj_pmid)

    os.makedirs(f"Words_PMID/Single_intersection/{short_path}", exist_ok=True)
    with open(f"Words_PMID/Single_intersection/{short_path}/Object_dict.json", "w") as f:
        json.dump(common_obj_pmid_dict, f)

    common_sub_pmid_dict = {}
    Tokenized_sub_dict = {}
    for sub in tqdm(Subject_list):
        sub = sub.lower()
        words = tokenizer.tokenize(sub)
        Tokenized_sub_dict[sub] = words

    for sub, words in tqdm(Tokenized_sub_dict.items()):
        common_sub_pmid = set(Subject_dict.get(words[0], []))
        for word in words[1:]:
            common_sub_pmid.intersection_update(Subject_dict.get(word, []))
        common_sub_pmid_dict[sub] = list(common_sub_pmid)

        # for idx, word in enumerate(words):
        #     if word not in Subject_dict.keys():
        #         continue
        #     word_pmid = set(Subject_dict[word])
        #     if idx == 0:
        #         common_sub_pmid = set(word_pmid)
        #     else:
        #         common_sub_pmid = common_sub_pmid.intersection(word_pmid)
        # common_sub_pmid_dict[sub] = list(common_sub_pmid)

    os.makedirs(f"Words_PMID/Single_intersection/{short_path}", exist_ok=True)
    with open(f"Words_PMID/Single_intersection/{short_path}/Subject_dict.json", "w") as f:
        json.dump(common_sub_pmid_dict, f)



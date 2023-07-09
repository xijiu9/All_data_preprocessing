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


Relation_path = list(set(glob.glob("Triple_result/BioLAMA/*/*/", recursive=True)))

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

with open('../Divide_Into_Months/data/refined_pubmed_abstract.json') as f:
    Pubmed_data = json.load(f)

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "finish loading Pubmed")

for file_path in Relation_path:
    short_path = "/".join(file_path.split("/")[-4:-1])
    print(short_path)

    with open(f'Triple_result/{short_path}/Triple_list.json') as f:
        Triple_list = json.load(f)

    object_path = f"Words_PMID/Single_intersection/{short_path}/Object_dict.json"
    subject_path = f"Words_PMID/Single_intersection/{short_path}/Subject_dict.json"
    with open(object_path) as f:
        Object_dict = json.load(f)
    with open(subject_path) as f:
        Subject_dict = json.load(f)

    exact_Object_dict = {key: [] for key in Object_dict.keys()}
    exact_Subject_dict = {key: [] for key in Subject_dict.keys()}

    for obj, search_field_relation_pmid in tqdm(Object_dict.items()):
        for pmid in search_field_relation_pmid:
            if Pubmed_data[pmid][0][-1] in ['.', '?', '!']:
                article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + Pubmed_data[pmid][1]
            else:
                article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + '. ' + Pubmed_data[pmid][1]
            article = article.lower()

            if obj in article:
                exact_Object_dict[obj].append(pmid)

    for sub, search_field_relation_pmid in tqdm(Subject_dict.items()):
        for pmid in search_field_relation_pmid:
            if Pubmed_data[pmid][0][-1] in ['.', '?', '!']:
                article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + Pubmed_data[pmid][1]
            else:
                article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + '. ' + Pubmed_data[pmid][1]
            article = article.lower()

            if sub in article:
                exact_Subject_dict[sub].append(pmid)


    os.makedirs(f"Words_PMID/Exact_intersection/{short_path}", exist_ok=True)
    with open(f"Words_PMID/Exact_intersection/{short_path}/Object_dict.json", "w") as f:
        json.dump(exact_Object_dict, f)

    os.makedirs(f"Words_PMID/Exact_intersection/{short_path}", exist_ok=True)
    with open(f"Words_PMID/Exact_intersection/{short_path}/Subject_dict.json", "w") as f:
        json.dump(exact_Subject_dict, f)


import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm
import pyfiglet


Relation_path = sorted(list(set(glob.glob("Extract Result/e_result/*/*/*/", recursive=True))))

tokenizer = nltk.RegexpTokenizer(r'[\w-]+')

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

with open('../Divide Into Months/data/refined_pubmed_abstract.json') as f:
    Pubmed_data = json.load(f)

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "finish loading Pubmed")

def flatten(lst):
    # print(11111)
    result = []
    for item in lst:
        if isinstance(item, str) and item.startswith('[') and item.endswith(']'):
            try:
                eval_lst = eval(item)
            except:
                print(item)
                eval_lst = [item[1:-1]]
                print(eval_lst)
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

    for triple in Triple_list:
        objs = triple["object"]
        obj_syns = triple["object_synonym"]

        objs = flatten(objs)
        obj_syns = flatten(obj_syns)

        objs.extend(obj_syns)

        subs = triple["subject"]
        sub_syns = triple["subject_synonym"]
        subs = flatten(subs)
        sub_syns = flatten(sub_syns)
        subs.extend(sub_syns)

        union_objs_pmid = {}
        for obj in objs:
            words = tokenizer.tokenize(obj)
            for idx, word in enumerate(words):
                word = word.lower()
                if word not in Object_dict.keys():
                    print(f"CAN NOT FIND {word}")
                    continue
                word_pmid = Object_dict[word]
                # print(word, word_pmid[:5], word_pmid[-5:])
                if idx == 0:
                    common_obj_pmid = set(word_pmid)
                else:
                    common_obj_pmid = common_obj_pmid.intersection(word_pmid)
                # print(list(common_obj_pmid)[:5], list(common_obj_pmid)[-5:])

            print("OBJ ", len(common_obj_pmid), obj)
            if len(common_obj_pmid):
                for pmid in common_obj_pmid:
                    article = Pubmed_data[pmid]
                    if obj in article:
                        union_objs_pmid.add(pmid)

        union_subs_pmid = {}
        for sub in subs:
            words = tokenizer.tokenize(sub)
            for idx, word in enumerate(words):
                word = word.lower()
                if word not in subect_dict.keys():
                    print(f"CAN NOT FIND {word}")
                    continue
                word_pmid = subect_dict[word]
                # print(word, word_pmid[:5], word_pmid[-5:])
                if idx == 0:
                    common_sub_pmid = set(word_pmid)
                else:
                    common_sub_pmid = common_sub_pmid.intersection(word_pmid)
                # print(list(common_sub_pmid)[:5], list(common_suu_pmid)[-5:])

            print("SUB ", len(common_sub_pmid), sub)
            if len(common_sub_pmid):
                for pmid in common_sub_pmid:
                    article = Pubmed_data[pmid]
                    if sub in article:
                        union_subs_pmid.add(pmid)

        union_relation_pmid = union_subs_pmid.intersection(union_objs_pmid)
        print("RELATION ", len(union_relation_pmid), union_relation_pmid[:5], union_relation_pmid[-5:])

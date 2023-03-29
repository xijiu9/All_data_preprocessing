import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm
import copy
import pyfiglet


Relation_path = sorted(list(set(glob.glob("Extract Result/e_result/BioLAMA/wikidata/*/", recursive=True))))

Relation_path.reverse()

tokenizer = nltk.RegexpTokenizer(r'[\w-]+')

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

with open('../Divide_Into_Months/data/refined_pubmed_abstract.json') as f:
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

    Triple_list_with_relation = []
    for triple in tqdm(Triple_list):

        objs = triple["object"]
        obj_syns = triple["object_synonym"]

        objs = flatten([objs])
        obj_syns = flatten([obj_syns])

        objs.extend(obj_syns)

        subs = triple["subject"]
        sub_syns = triple["subject_synonym"]

        subs = flatten([subs])
        sub_syns = flatten([sub_syns])

        subs.extend(sub_syns)

        # object
        union_objs_pmid = set([])
        union_objs_dict = {}
        for obj in objs:
            obj = obj.lower()
            words = tokenizer.tokenize(obj)
            for idx, word in enumerate(words):
                word = word.lower()
                if word not in Object_dict.keys():
                    # print(f"CAN NOT FIND {word}")
                    continue
                word_pmid = Object_dict[word]
                # print(word, word_pmid[:5], word_pmid[-5:])
                if idx == 0:
                    common_obj_pmid = set(word_pmid)
                else:
                    common_obj_pmid = common_obj_pmid.intersection(word_pmid)

            actual_common_obj_pmid = set([])
            if len(common_obj_pmid):
                for pmid in common_obj_pmid:
                    if Pubmed_data[pmid][0][-1] in ['.', '?', '!']:
                        article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + Pubmed_data[pmid][1]
                    else:
                        article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + '. ' + Pubmed_data[pmid][1]
                    article = article.lower()

                    if obj in article:
                        union_objs_pmid.add(pmid)
                        actual_common_obj_pmid.add(pmid)
                        union_objs_dict[pmid] = obj

            # if len(actual_common_obj_pmid):
            #     print("OBJ ", obj)
            #     print("OBJ ", len(common_obj_pmid), list(common_obj_pmid)[:5], list(common_obj_pmid)[-5:])
            #     print("OBJ ", len(actual_common_obj_pmid), list(actual_common_obj_pmid)[:5], list(actual_common_obj_pmid)[-5:])

        union_subs_pmid = set([])
        union_subs_dict = {}
        for sub in subs:
            sub = sub.lower()
            words = tokenizer.tokenize(sub)
            for idx, word in enumerate(words):
                # print(word)
                word = word.lower()
                if word not in Subject_dict.keys():
                    # print(f"CAN NOT FIND {word}")
                    continue
                word_pmid = Subject_dict[word]
                # print(word, word_pmid[:5], word_pmid[-5:])
                if idx == 0:
                    common_sub_pmid = set(word_pmid)
                else:
                    common_sub_pmid = common_sub_pmid.intersection(word_pmid)
                # print(list(common_sub_pmid)[:5], list(common_suu_pmid)[-5:])

            actual_common_sub_pmid = set([])
            if len(common_sub_pmid):
                for pmid in common_sub_pmid:
                    if Pubmed_data[pmid][0][-1] in ['.', '?', '!']:
                        article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + Pubmed_data[pmid][1]
                    else:
                        article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + '. ' + Pubmed_data[pmid][1]
                    article = article.lower()

                    if sub in article:
                        union_subs_pmid.add(pmid)
                        actual_common_sub_pmid.add(pmid)
                        union_subs_dict[pmid] = sub

            # if len(actual_common_sub_pmid):
            #     print("SUB ", sub)
            #     print("SUB ", len(common_sub_pmid), list(common_sub_pmid)[:5], list(common_sub_pmid)[-5:])
            #     print("SUB ", len(actual_common_sub_pmid), list(actual_common_sub_pmid)[:5], list(actual_common_sub_pmid)[-5:])

        union_relation_pmid = set(union_subs_pmid).intersection(set(union_objs_pmid))

        # print("RELATION ", len(union_relation_pmid), list(union_relation_pmid)[:5], list(union_relation_pmid)[-5:])
        if len(union_relation_pmid):
            # print(objs[0] + "  " + subs[0])

            new_triple = triple.copy()
            new_triple["extract relation pairs"] = {}
            for pmid in union_relation_pmid:
                new_triple["extract relation pairs"][pmid] = {"sub": union_subs_dict[pmid],
                                                              "obj": union_objs_dict[pmid]}
            new_triple["extract relation pairs"] = dict(sorted(new_triple["extract relation pairs"].items()))
            # print(new_triple["extract relation pairs"])

            Triple_list_with_relation.append(new_triple)
            os.makedirs(os.path.join("Triple with Relation", short_path), exist_ok=True)
            with open(os.path.join("Triple with Relation", short_path, "New_Triple_list.json"), "w") as f:
                json.dump(Triple_list_with_relation, f, indent=4)




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

Relation_path = sorted(list(set(glob.glob("Triple_result/MedLAMA/*/*/", recursive=True))))

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

    Triple_list_with_relation = []
    for triple in tqdm(Triple_list):
        
        # if triple["subject_index"] != "C4727684":
        #     continue

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

        objs = [obj.lower() for obj in objs]
        subs = [sub.lower() for sub in subs]

        search_field_objs_pmid = set([])
        search_field_objs_dict = {}
        for obj in objs:
            for pmid in Object_dict[obj]:
                search_field_objs_pmid.add(pmid)

                if pmid not in search_field_objs_dict.keys():
                    search_field_objs_dict[pmid] = [obj, ]
                else:
                    search_field_objs_dict[pmid].append(obj)

        search_field_subs_pmid = set([])
        search_field_subs_dict = {}
        for sub in subs:
            for pmid in Subject_dict[sub]:
                search_field_subs_pmid.add(pmid)

                if pmid not in search_field_subs_dict.keys():
                    search_field_subs_dict[pmid] = [sub, ]
                else:
                    search_field_subs_dict[pmid].append(sub)

        search_field_relation_pmid = set(search_field_subs_pmid).intersection(set(search_field_objs_pmid))

        # print(search_field_relation_pmid)
        # IPython.embed()

        union_objs_pmid = set([])
        union_objs_dict = {}
        for pmid in search_field_relation_pmid:
            objs = search_field_objs_dict[pmid]
            for obj in objs:
                if Pubmed_data[pmid][0][-1] in ['.', '?', '!']:
                    article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + Pubmed_data[pmid][1]
                else:
                    article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + '. ' + Pubmed_data[pmid][1]
                article = article.lower()

                if obj in article:
                    union_objs_pmid.add(pmid)

                    if pmid not in union_objs_dict.keys():
                        union_objs_dict[pmid] = [obj, ]
                    else:
                        union_objs_dict[pmid].append(obj)

        union_subs_pmid = set([])
        union_subs_dict = {}
        for pmid in search_field_relation_pmid:
            subs = search_field_subs_dict[pmid]
            for sub in subs:
                if Pubmed_data[pmid][0][-1] in ['.', '?', '!']:
                    article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + Pubmed_data[pmid][1]
                else:
                    article = Pubmed_data[pmid][0].replace('[', '').replace(']', '') + '. ' + Pubmed_data[pmid][1]
                article = article.lower()

                if sub in article:
                    union_subs_pmid.add(pmid)

                    if pmid not in union_subs_dict.keys():
                        union_subs_dict[pmid] = [sub, ]
                    else:
                        union_subs_dict[pmid].append(sub)

        union_relation_pmid = set(union_subs_pmid).intersection(set(union_objs_pmid))
        # print(union_relation_pmid)

        # IPython.embed()

        if len(union_relation_pmid):
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

        # if triple["subject_index"] == "C4727684":
        #     break

    print(f"Have {len(Triple_list_with_relation)} relations now.")


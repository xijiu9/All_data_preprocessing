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

# print(time.asctime(time.localtime()))
#
# data_path = "/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Divide_Into_Months/data/pubmed_abstract.json"
# with open(data_path) as f:
#     Pubmed_Data = json.load(f)
#
# print(time.asctime(time.localtime()))
#
# IPython.embed()


# print(time.asctime(time.localtime()))
#
# data_path = "/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Triple with Relation/BioLAMA/ctd/C/New_Triple_list.json"
# with open(data_path) as f:
#     Data = json.load(f)
#
# print(len(Data))
#
# for triple in Data:
#     if triple["subject_index"] == "C118874":
#         print(triple)


# object_path = "/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Single_intersection/" \
#               "BioLAMA/ctd/CD1/Object_dict.json"
# subject_path = "/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Single_intersection/" \
#                "BioLAMA/ctd/CD1/Subject_dict.json"
# with open(object_path) as f:
#     Object_dict = json.load(f)
# with open(subject_path) as f:
#     Subject_dict = json.load(f)
# print(len(Object_dict.keys()))
# print(len(Subject_dict.keys()))
# print(time.asctime(time.localtime()))
#
# Obj_keys, Sub_keys = list(Object_dict.keys()), list(Subject_dict.keys())
#
# print(Obj_keys[:3])
# print(Sub_keys[:3])
#
# print(len(Subject_dict["TAK-875".lower()]))
# print(len(Object_dict["Type 2 Diabetes".lower()]))
#
# SS, OO = Subject_dict["TAK-875".lower()], Object_dict["Type 2 Diabetes".lower()]
#
# common_SO = set(SS).intersection(set(OO))
# print(common_SO)
#
# IPython.embed()
#
# cnt = 0
# for sub, pmids in Subject_dict.items():
#     pmid_set = set(pmids)
#
#     if pmid_set.intersection(set(OO)):
#         cnt += 1
#
# print(cnt)



########### use for determine test_Obj and test_Sub
# print(time.asctime(time.localtime()))
#
# object_path = "/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Single_intersection/MedLAMA/medlama/may_treat_1000/Object_dict.json"
# subject_path = "/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Single_intersection/MedLAMA/medlama/may_treat_1000/Subject_dict.json"
# with open(object_path) as f:
#     Object_dict = json.load(f)
# with open(subject_path) as f:
#     Subject_dict = json.load(f)
# print(time.asctime(time.localtime()))
#
# pcos = Object_dict["parkinson disease".lower()]
# deso = Subject_dict["safinamide mesylate".lower()]
#
# print(pcos)
# print(deso)
# print(len(pcos))
# print(len(deso))
#
# # IPython.embed()
# print(set(pcos).intersection(set(deso)))
#
# # IPython.embed()
#
# Obj_time_dict, Sub_time_dict = {}, {}
# for key in tqdm(pcos):
#     key = int(key)
#     if key < 25000000:
#         continue
#
#     try:
#         with open(f"Fetch_Date_Info/data_map/{5000 * (key // 5000)}-{5000 * (key // 5000 + 1)}.json") as f:
#             time_map = json.load(f)
#     except:
#         print("wrong")
#         print(f"Fetch_Date_Info/data_map/{5000 * (key // 5000)}-{5000 * (key // 5000 + 1)}.json")
#         continue
#
#     year, month, day = int(time_map[f"{key}"]["full"]["EDAT"][0:4]), int(time_map[f"{key}"]["full"]["EDAT"][5:7]), int(
#         time_map[f"{key}"]["full"]["EDAT"][8:10])
#     full_time = time_map[f"{key}"]["full"]["EDAT"]
#
#     Obj_time_dict[key] = full_time
#
# with open("test_Obj.json", "w") as f:
#     json.dump(Obj_time_dict, f, indent=4)
#
# for key in tqdm(deso):
#     key = int(key)
#     try:
#         with open(f"Fetch_Date_Info/data_map/{5000 * (key // 5000)}-{5000 * (key // 5000 + 1)}.json") as f:
#             time_map = json.load(f)
#     except:
#         print("wrong")
#         print(f"Fetch_Date_Info/data_map/{5000 * (key // 5000)}-{5000 * (key // 5000 + 1)}.json")
#         continue
#
#     year, month, day = int(time_map[f"{key}"]["full"]["EDAT"][0:4]), int(time_map[f"{key}"]["full"]["EDAT"][5:7]), int(
#         time_map[f"{key}"]["full"]["EDAT"][8:10])
#     full_time = time_map[f"{key}"]["full"]["EDAT"]
#
#     Sub_time_dict[key] = full_time
#
# with open("test_Sub.json", "w") as f:
#     new_triple_list = json.dump(Sub_time_dict, f, indent=4)



#
# import IPython
# IPython.embed()

#
# triples_with_relation_path = sorted(glob.glob("Extract Relation/Triple with Relation/*/*/*/New_Triple_list.json"))
#
# Triple_list_after_2020 = []
#
# all_triples, all_after_2023_triples = 0, 0
# for file_path in triples_with_relation_path:
#     with open(file_path) as f:
#         new_triple_list = json.load(f)
#
#     print(file_path)
#     print(len(new_triple_list))
#     all_triples += len(new_triple_list)
#
#     part_triples_after_2023 = 0
#
#     for triple in new_triple_list:
#         Trikeys = triple["extract relation pairs"].keys()
#         Trikeys = [int(num) for num in Trikeys]
#         if np.min(np.array(Trikeys)) > 28000000:
#             # print(1)
#             all_after_2023_triples += 1
#             part_triples_after_2023 += 1
#
#             Triple_list_after_2020.append(triple)
#
#     print(part_triples_after_2023)
#
# print(all_triples, all_after_2023_triples)
#
# with open("Triple_list_after_2020") as f:
#     new_triple_list = json.dump(Triple_list_after_2020, f)


print(time.asctime(time.localtime()))

with open('Extract_Relation/Extract Result/e_result/MedLAMA/medlama/may_treat_1000/Subject_dict.json') as f:
    Subject_dict = json.load(f)

print(time.asctime(time.localtime()))

IPython.embed()
#
# values = []
# for k, v in Object_dict.items():
#     values.append(len(v))
#
# # Plot the histogram
# plt.hist(values, bins=100, log=True)
# plt.savefig("test.png")
#
# sorted_dict = dict(sorted(Object_dict.items(), key=lambda item: len(item[1])))
#
# for k ,v in sorted_dict.items():
#     print(k, len(v))
#
# with open('data.txt', 'w') as f:
#     for k, v in sorted_dict.items():
#         f.write(f'{k}: {len(v)}\n')

# print(len(sorted_dict["cancer"]))
# # print(len(sorted_dict["of"]))
# # print(len(sorted_dict["the"]))
# print(len(sorted_dict["ovary"]))
#
# print(sorted_dict["cancer"][:5])
# print(sorted_dict["ovary"][:5])
#
# common_elements = set(sorted_dict["cancer"]).intersection(set(sorted_dict["ovary"]))
#
# print(len(common_elements))
# print(common_elements)







# tokenizer = nltk.RegexpTokenizer(r'[\w-]+')
#
# with open('Extract_relation/data/BioLAMA/ctd/CD1/Object_list.json') as f:
#     Object_list = json.load(f)
#
# print(len(Object_list))
#
# Object_set = set(Object_list)
# print(len(Object_set))
#
# Object_set_single = set()
# for string in Object_set:
#     words = tokenizer.tokenize(string)
#     Object_set_single.update(words)
#
# print(len(Object_set_single))
#
#
# Object_set_lower = {s.lower() for s in Object_set}
# Object_set_single_lower = set()
# for string in Object_set_lower:
#     words = tokenizer.tokenize(string)
#     Object_set_single_lower.update(words)
#
# print(len(Object_set_single_lower))
#
# diff1 = Object_set - Object_set_single
# # print(diff1)
# # print("Myokymia" in Object_set_single)
#
# diff2 = Object_set_single - Object_set
# # print(diff2)
#
# # for i in Object_list:
# #     if "13" in i:
# #         print(i)
#
# duplicates = set()
#
# test_lower = set()
#
# for word in Object_set_single:
#     lowercase_word = word.lower()
#     if lowercase_word in test_lower:
#         duplicates.add(lowercase_word)
#     else:
#         test_lower.add(lowercase_word)
#
# print("The following words are duplicates:")
# for duplicate in duplicates:
#     for word in Object_set_single:
#         if word.lower() == duplicate:
#             print(word)
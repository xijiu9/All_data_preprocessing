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

triples_with_relation_path = sorted(glob.glob("Extract Relation/Triple with Relation/*/*/*/New_Triple_list.json"))

Triple_list_after_2020 = []

all_triples, all_after_2023_triples = 0, 0
for file_path in triples_with_relation_path:
    with open(file_path) as f:
        new_triple_list = json.load(f)

    print(file_path)
    print(len(new_triple_list))
    all_triples += len(new_triple_list)

    part_triples_after_2023 = 0

    for triple in new_triple_list:
        Trikeys = triple["extract relation pairs"].keys()
        Trikeys = [int(num) for num in Trikeys]
        if np.min(np.array(Trikeys)) > 28000000:
            # print(1)
            all_after_2023_triples += 1
            part_triples_after_2023 += 1

            Triple_list_after_2020.append(triple)

    print(part_triples_after_2023)

print(all_triples, all_after_2023_triples)

with open("Triple_list_after_2020") as f:
    new_triple_list = json.dump(Triple_list_after_2020, f)


# print(time.asctime(time.localtime()))
#
# with open('Extract Relation/Extract Result/e_result/BioLAMA/ctd/CD1/Object_dict.json') as f:
#     Object_dict = json.load(f)
#
# print(time.asctime(time.localtime()))
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
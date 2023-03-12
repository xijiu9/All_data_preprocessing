import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm
import matplotlib.pyplot as plt
import IPython

print(time.asctime(time.localtime()))

with open('Extract_relation/Extract Result/BioLAMA/ctd/CD1/Object_dict.json') as f:
    Object_dict = json.load(f)

print(time.asctime(time.localtime()))

values = []
for k, v in Object_dict.items():
    values.append(len(v))

# Plot the histogram
plt.hist(values, bins=100, log=True)
plt.savefig("test.png")

sorted_dict = dict(sorted(Object_dict.items(), key=lambda item: len(item[1])))

for k ,v in sorted_dict.items():
    print(k, len(v))



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
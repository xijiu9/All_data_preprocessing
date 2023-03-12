import numpy as np   # For numerical computing and array manipulation
import requests  # For making HTTP requests
import json  # For working with JSON data
import os  # For interacting with the operating system
import time
import IPython

new_article_list = {}
discard_article_list = {}

print(time.asctime(time.localtime()))
with open('data/pubmed_abstract.json', 'r') as f:
    pubmed_data = json.load(f)
print(time.asctime(time.localtime()), "Finish Pubmed")

# print(pubmed_data["535000"])

def overlap(str1, str2, special=False):
    str1, str2 = str1.lower(), str2.lower()
    words1 = str1.split()
    words2 = str2.split()
    common = set(words1) & set(words2)
    num_common = len(common)

    if special:
        IPython.embed()

    if min(len(words1), len(words2)) == 0:
        return 0
    else:
        return num_common / min(len(words1), len(words2))

for file_name in sorted(os.listdir("../Fetch Date Info/data_map")):
# for file_name in ["20670000-20675000.json"]:
    data_map = json.load(open("../Fetch Date Info/data_map/{}".format(file_name), 'r'))

    for pmid in data_map.keys():

        if pmid not in pubmed_data.keys() or pubmed_data[pmid][0] in ['', None]:
            continue
        overlap_rate1 = overlap(data_map[pmid]["title broken"], pubmed_data[pmid][0])
        if "AB" in data_map[pmid]["full"].keys():
            overlap_rate2 = overlap(data_map[pmid]["full"]["AB"], pubmed_data[pmid][1])
        else:
            overlap_rate2 = 0

        if overlap_rate1 < 0.5 and overlap_rate2 < 0.7:
            discard_article_list[pmid] = {}
            discard_article_list[pmid]["data_map_info"] = data_map[pmid]
            discard_article_list[pmid]["pubmed_offset_info"] = {pmid: pubmed_data[pmid]}

            print(pmid)
            # print(data_map[pmid]["title broken"], "|" * 10, pubmed_data[pmid][0])
            # print(pubmed_data[pmid][1])
            continue
        new_article_list[pmid] = pubmed_data[pmid]

print("Finish, start to save data")
with open('data/refined_pubmed_abstract.json', 'w') as f:
    json.dump(new_article_list, f)

with open('data/discard_pubmed_abstract.json', 'w') as f:
    json.dump(discard_article_list, f, indent=4)


print(len(new_article_list), len(pubmed_data), len(discard_article_list))




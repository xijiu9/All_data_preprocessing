import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "Start Loading")

BioLAMA_files = sorted(glob.glob("Triple_result/BioLAMA/*/*/", recursive=True))
MedLAMA_files = sorted(glob.glob("Triple_result/MedLAMA/*/*/", recursive=True))


BioLAMA_files.extend(MedLAMA_files)

BioMedLAMA_files = [dir for dir in BioLAMA_files if os.path.isdir(dir)]

print(BioMedLAMA_files)

with open('../DIVIDE INTO MONTHS/data/refined_pubmed_abstract.json') as f:
    Pubmed_data = json.load(f)

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "finish loading Pubmed")

tokenizer = nltk.RegexpTokenizer(r'[\w-]+')

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

Object_all_dict = {}
Subject_all_dict = {}

for file_path in BioMedLAMA_files:
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), f" {file_path}")

    with open(f'{file_path}/Object_list.json') as f:
        Object_list = json.load(f)

    with open(f'{file_path}/Subject_list.json') as f:
        Subject_list = json.load(f)

    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "finish loading Relation")

    Object_set = set(Object_list)
    Object_set = {s.lower() for s in Object_set}
    Object_set_single = set()
    for string in Object_set:
        words = tokenizer.tokenize(string)
        Object_set_single.update(words)

    Subject_set = set(Subject_list)
    Subject_set = {s.lower() for s in Subject_set}
    Subject_set_single = set()
    for string in Subject_set:
        words = tokenizer.tokenize(string)
        Subject_set_single.update(words)

    Object_set_single = Object_set_single - stop_words
    Subject_set_single = Subject_set_single - stop_words

    Object_dict = {k: [] for k in Object_set_single}
    Subject_dict = {k: [] for k in Subject_set_single}


    print(list(Object_set_single)[:10])
    print(list(Subject_set_single)[:10])

    print(len(Object_list), len(Object_set_single), len(Object_dict))
    print(len(Subject_list), len(Subject_set_single), len(Subject_dict))

    Disjoint_Object, Joint_Object, Disjoint_Subject, Joint_Subject = 0, 0, 0, 0

    # for pmid in Pubmed_data.keys():
    for pmid in tqdm(Pubmed_data.keys(), desc="Set Disjoint of Pubmed Data and Relation Triples"):
        tokenized_article = set(tokenizer.tokenize(' '.join(Pubmed_data[pmid])))
        tokenized_article = {s.lower() for s in tokenized_article}

        Object_common = tokenized_article.intersection(Object_set_single)
        Subject_common = tokenized_article.intersection(Subject_set_single)

        for obj in Object_common:
            Object_dict[obj].append(pmid)
        for sub in Subject_common:
            Subject_dict[sub].append(pmid)


        # if tokenized_article.isdisjoint(Object_set_single):
        #     Disjoint_Object += 1
        # else:
        #     Joint_Object += 1
        #
        # if tokenized_article.isdisjoint(Subject_set_single):
        #     Disjoint_Subject += 1
        # else:
        #     Joint_Subject += 1

    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "finish Processing Pubmed")

    if not os.path.exists("Extract Result"):
        os.makedirs("Extract Result")

    # Save Subject dictionary to JSON file
    Subject_dict_filename = os.path.join("Extract Result", file_path[5:])
    os.makedirs(Subject_dict_filename, exist_ok=True)
    with open(os.path.join(Subject_dict_filename, "Subject_dict.json"), "w") as f:
        json.dump(Subject_dict, f)

    # Save Object dictionary to JSON file
    Object_dict_filename = os.path.join("Extract Result", file_path[5:])
    os.makedirs(Object_dict_filename, exist_ok=True)
    with open(os.path.join(Object_dict_filename, "Object_dict.json"), "w") as f:
        json.dump(Object_dict, f)

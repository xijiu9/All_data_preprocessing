import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm

def create_triple(data_belong):
    triple = {
        "data_belong": data_belong,
        "data_path": "",
        "data_short": "",
        "subject": "",
        "subject_synonym": "",
        "subject_index": "",
        "relation": "",
        "relation_prompt": "",
        "object": "",
        "object_synonym": "",
        "object_index": ""
    }
    return triple

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

Triple_dict = {}


# BioLAMA
Prompt_dict = {}

BioLAMA_path = sorted(glob.glob("../../BioLAMA/data/**/triples_processed/**/*.jsonl", recursive=True))
Prompt_path = sorted(glob.glob("../../BioLAMA/data/**/prompts/manual.jsonl", recursive=True))
for file_path in Prompt_path:
    with open(file_path, 'r') as f:
        for line in f:
            prompt = json.loads(line)
            Prompt_dict[prompt["relation"]] = prompt["template"]

for file_path in BioLAMA_path:
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            triple = create_triple("BioLAMA")

            triple['data_path'] = file_path

            dir_parts = os.path.dirname(file_path).split('/')
            data_short = '/'.join([dir_parts[2], dir_parts[4], dir_parts[6]])
            triple['data_short'] = data_short

            triple['subject'] = data['sub_label']
            triple['subject_synonym'] = data['sub_aliases']
            triple['subject_index'] = data['sub_uri']

            triple['relation'] = data['predicate_id']
            triple['relation_prompt'] = Prompt_dict[data['predicate_id']]

            triple['object'] = data['obj_labels']
            triple['object_synonym'] = data['obj_aliases']
            triple['object_index'] = data['obj_uris']

            if data_short not in Triple_dict.keys():
                Triple_dict[data_short] = [triple, ]
            else:
                Triple_dict[data_short].append(triple)

    #         break
    # break

# MedLAMA
Prompt_dict = {}

MedLAMA_path = sorted(glob.glob("../../MedLAMA/data/medlama/2021AA/*.csv", recursive=True))
Prompt_path = sorted(glob.glob("../../MedLAMA/data/medlama/prompts.csv", recursive=True))
for file_path in Prompt_path:
    prompts = pd.read_csv(file_path)
    for index, prompt in prompts.iterrows():
        Prompt_dict[prompt["pid"]] = prompt["human_prompt"]

for file_path in MedLAMA_path:
    df = pd.read_csv(file_path)
    for index, data in df.iterrows():
        triple = create_triple("MedLAMA")

        triple['data_path'] = file_path

        dir_parts = os.path.dirname(file_path).split('/')
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        data_short = '/'.join([dir_parts[2], dir_parts[4], file_name])
        triple['data_short'] = data_short

        triple['subject'] = data['head_name']
        triple['subject_synonym'] = None
        triple['subject_index'] = data['head_cui']

        triple['relation'] = data['rel']
        triple['relation_prompt'] = Prompt_dict[data['rel']]

        triple['object'] = data['tail_names_list']
        triple['object_synonym'] = None
        triple['object_index'] = data['tail_cuis_list']

        if data_short not in Triple_dict.keys():
            Triple_dict[data_short] = [triple, ]
        else:
            Triple_dict[data_short].append(triple)

    #     break
    # break


for data_short, Triple_list in Triple_dict.items():
    save_dir = f"Triple_result/{data_short}"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "Triple_list.json"), "w") as f:
        json.dump(Triple_list, f, indent=4)

    Object_list, Subject_list = [], []
    for triple in Triple_list:
        Object_list.append(triple['object'])
        Object_list.append(triple['object_synonym'])

        Subject_list.append(triple['subject'])
        Subject_list.append(triple['subject_synonym'])

    print(f"{data_short} original have {len(Subject_list)} Subject and {len(Object_list)} Object")
    Object_list = flatten(Object_list)
    Subject_list = flatten(Subject_list)

    Object_list = set(Object_list)
    Subject_list = set(Subject_list)
    print(f"{data_short} Set have {len(Subject_list)} Subject and {len(Object_list)} Object")

    with open(os.path.join(save_dir, "Object_list.json"), "w") as f:
        json.dump(list(Object_list), f, indent=4)
    with open(os.path.join(save_dir, "Subject_list.json"), "w") as f:
        json.dump(list(Subject_list), f, indent=4)
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

Subject_path = sorted(glob.glob("../Extract Relation/Triple_result/*/*/*/Subject_list.json"))
Object_path = sorted(glob.glob("../Extract Relation/Triple_result/*/*/*/Object_list.json"))

Subject_list, Object_list = [], []
for sub_path, obj_path in zip(Subject_path, Object_path):
    print(sub_path)
    with open(sub_path, "r") as f:
        sub_list = json.load(f)
    with open(obj_path, "r") as f:
        obj_list = json.load(f)

    Subject_list.extend(sub_list)
    Object_list.extend(obj_list)

print(len(Subject_list))
print(len(Object_list))

Subject_list, Object_list = list(set(Subject_list)), list(set(Object_list))

print(len(Subject_list))
print(len(Object_list))

os.makedirs("Summarize", exist_ok=True)
with open("Summarize/Subject_list.json", "w") as f:
    json.dump(Subject_list, f, indent=4)
with open("Summarize/Object_list.json", "w") as f:
    json.dump(Object_list, f, indent=4)



import json
import time
import numpy as np

from tqdm import tqdm, trange

# # # # 读取ALL_Object_dict.json
with open("/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Exact_intersection/ALL_Object_dict.json") as f:
    Object_dict = json.load(f)

# 读取ALL_Subject_dict.json
with open("/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Exact_intersection/ALL_Subject_dict.json") as f:
    Subject_dict = json.load(f)
#
# import IPython
# IPython.embed()
#
# print("finish loading")
# print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")
#
Object_dict_filtered = {key: value for key, value in Object_dict.items() if len(value) > 0}
Subject_dict_filtered = {key: value for key, value in Subject_dict.items() if len(value) > 0}

import IPython
IPython.embed()
import argparse
import csv
import os
import re
import time

from itertools import groupby
from lxml.builder import E
from lxml.etree import tostring
import tqdm

import time
import pickle
import numpy as np
import dump
import json
#
# input_path = "/m-ent1/ent1/xihc20/bioconcepts2pubtatorcentral.offset"
# # 7min 5188693 / s
# fo = open(input_path, "r")
# cnt = 0
# print(time.asctime(time.localtime()))
#
# keyword_list = {}
# for line in fo:
# # for line in tqdm.tqdm(fo):
#     if "|t|" in line:
#         cutline = line.split('|')
#         save_int, save_title = cutline[0], cutline[-1]
#         keyword_list[save_int] = []
#     elif "|a|" in line:
#         cutline = line.split('|')
#         if len(cutline[-1]) > 1e7 or len(cutline[-1]) < 10:
#             continue
#     else:
#         if line != '\n':
#             split_data = line.split('\t')
#             try:
#                 keyword = split_data[3].lower()
#             except:
#                 print(line)
#             if keyword not in keyword_list[save_int]:
#                 keyword_list[save_int].append(keyword)
#
# print(time.asctime(time.localtime()))
#
# with open('data/biomedical_word_list.json', 'w') as f:
#     json.dump(keyword_list, f)

with open('data/biomedical_word_list.json', 'r') as f:
    keyword_list = json.load(f)

import IPython
IPython.embed()

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

input_path = "/m-ent1/ent1/xihc20/bioconcepts2pubtatorcentral.offset"
# 7min 5188693 / s
fo = open(input_path, "r")
cnt = 0
print(time.asctime(time.localtime()))

article_list = {}
for line in fo:
# for line in tqdm.tqdm(fo):
    if "|t|" in line:
        cutline = line.split('|')
        save_int, save_title = cutline[0], cutline[-1]

    if "|a|" in line:
        cutline = line.split('|')
        if len(cutline[-1]) > 1e7 or len(cutline[-1]) < 10:
            continue
        if cutline[0] == save_int:
             article_list[save_int] = [save_title.rstrip(), cutline[-1].rstrip()]

        else:
            print("not right")

print(time.asctime(time.localtime()))

with open('data/pubmed_abstract.json', 'w') as f:
    json.dump(article_list, f)
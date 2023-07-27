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

with open("pmids_in_q_2022_04.json", "r") as f:
    kk_pmid = json.load(f)
print(len(kk_pmid))
# kk_pmid = list(set(kk_pmid))
print(len(kk_pmid))

with open('data/refined_pubmed_abstract.json', 'r') as f:
    pubmed_data = json.load(f)

print("load over")

for key in kk_pmid:
    with open("data/pmids_in_q_2022_04.txt", "a") as fp:
        if pubmed_data[str(key)][0][-1] in ['.', '?', '!']:
            combine_data = pubmed_data[key][0].replace('[', '').replace(']', '') + pubmed_data[key][1]
        else:
            combine_data = pubmed_data[key][0].replace('[', '').replace(']', '') + '.' + pubmed_data[key][1]

        fp.write(combine_data)
        fp.write('\n')

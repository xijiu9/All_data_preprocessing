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

with open('pmids_in_q_2020_02.json', 'r') as f:
    pmid_data = json.load(f)

for pmid in tqdm.tqdm(pmid_data):
    with open(f"../Fetch_Date_Info/data_map/{5000 * (int(pmid) // 5000)}-{5000 * (int(pmid) // 5000 + 1)}.json") as f:
        time_map = json.load(f)

    value = time_map[pmid]["time"]
    year, month, day = value[0:4], value[5:7], value[8:10]
    if year != '2020' or not '04' <= month <= '06':
        print("!" * 1000, pmid, year, month, day)
    else:
        # pass
        print(pmid, year, month, day)
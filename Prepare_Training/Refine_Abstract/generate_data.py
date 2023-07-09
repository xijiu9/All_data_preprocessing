import argparse
import csv
import os
import re
import time

from itertools import groupby
from lxml.builder import E
from lxml.etree import tostring
from tqdm import tqdm

import time
import pickle
import numpy as np
import dump
import json


with open('../../Divide_Into_Months/data/refined_pubmed_abstract.json', 'r') as f:
    pubmed_data = json.load(f)

print("load over")

time_list_const = ['q_2019', 'q_2020_1', 'q_2020_2', 'q_2020_3', 'q_2020_4', 'q_2021_1', 'q_2021_2', 'q_2021_3', 'q_2021_4', 'q_2022_1', 'q_2022_2', 'q_2022_3', 'q_2022_4']

for time_before, time_after in zip(time_list_const[:-1], time_list_const[1:]):
    print(f"start loading {time_before}, {time_after}")
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

    # every 3 months
    date_path = f"{time_after}"

    with open(f"{time_after}_pmids.json", "r") as f:
        pmid_list = json.load(f)

    for pmid in tqdm(pmid_list):
        with open("{}_data.txt".format(date_path), "a") as fp:
            if pubmed_data[pmid][0][-1] in ['.', '?', '!']:
                combine_data = pubmed_data[pmid][0].replace('[', '').replace(']', '') + pubmed_data[pmid][1]
            else:
                combine_data = pubmed_data[pmid][0].replace('[', '').replace(']', '') + '.' + pubmed_data[pmid][1]

            fp.write(combine_data)
            fp.write('\n')

    del pmid_list
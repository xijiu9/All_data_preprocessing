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

with open("covid_pmid_map.json", "r") as f:
    covid_time_map = json.load(f)

print("load over")

time_list_const = ['q_2019', 'q_2020_1', 'q_2020_2', 'q_2020_3', 'q_2020_4', 'q_2021_1', 'q_2021_2', 'q_2021_3', 'q_2021_4', 'q_2022_1', 'q_2022_2', 'q_2022_3', 'q_2022_4']

for qtime in covid_time_map:
    print(f"start loading {qtime}")
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

    # every 3 months
    date_path = f"{qtime}"

    for pmid in tqdm(covid_time_map[f"{qtime}"]):
        with open("{}_data.txt".format(date_path), "a") as fp:
            if pmid not in pubmed_data.keys():
                import IPython
                IPython.embed()

            if pubmed_data[pmid][0][-1] in ['.', '?', '!']:
                combine_data = pubmed_data[pmid][0].replace('[', '').replace(']', '') + pubmed_data[pmid][1]
            else:
                combine_data = pubmed_data[pmid][0].replace('[', '').replace(']', '') + '.' + pubmed_data[pmid][1]

            fp.write(combine_data)
            fp.write('\n')

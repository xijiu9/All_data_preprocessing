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


with open('data/refined_pubmed_abstract.json', 'r') as f:
    pubmed_data = json.load(f)

print("load over")

for file_name in sorted(os.listdir("../Fetch Date Info/data_map")):
# for file_name in ["20670000-20675000.json"]:
    print(file_name)
    data_map = json.load(open("../Fetch Date Info/data_map/{}".format(file_name), 'r'))

    for key in data_map.keys():
        if key not in pubmed_data.keys():
            continue

        try:
            a = pubmed_data[key][0][-1]
        except:
            print(pubmed_data[key])
            print(key)

        value = data_map[key]["time"]
        year, month, day = value[0:4], value[5:7], value[8:10]
        date_path = "training_data/mode_y/y={}".format(year)
        os.makedirs(date_path, exist_ok=True)

        # every year
        with open("training_data/mode_y/y={}/data.txt".format(year), "a") as fp:
            if pubmed_data[key][0][-1] in ['.', '?', '!']:
                combine_data = pubmed_data[key][0].replace('[', '').replace(']', '') + pubmed_data[key][1]
            else:
                combine_data = pubmed_data[key][0].replace('[', '').replace(']', '') + '. ' + pubmed_data[key][1]

            fp.write(combine_data)
            fp.write('\n')

        # every month
        date_path = "training_data/mode_m/y={}/m={:02}".format(year, int(month))
        os.makedirs(date_path, exist_ok=True)
        with open("{}/data.txt".format(date_path), "a") as fp:
            if pubmed_data[key][-1] in ['.', '?', '!']:
                combine_data = pubmed_data[key][0].replace('[', '').replace(']', '') + pubmed_data[key][1]
            else:
                combine_data = pubmed_data[key][0].replace('[', '').replace(']', '') + '.' + pubmed_data[key][1]

            fp.write(combine_data)
            fp.write('\n')

        # every 3 months
        date_path = "training_data/mode_q/y={}/q={:02}".format(year, (int(month) - 1) // 3 + 1)
        os.makedirs(date_path, exist_ok=True)
        with open("{}/data.txt".format(date_path), "a") as fp:
            if pubmed_data[str(key)][0][-1] in ['.', '?', '!']:
                combine_data = pubmed_data[key][0].replace('[', '').replace(']', '') + pubmed_data[key][1]
            else:
                combine_data = pubmed_data[key][0].replace('[', '').replace(']', '') + '.' + pubmed_data[key][1]

            fp.write(combine_data)
            fp.write('\n')

        # every 6 months
        date_path = "training_data/mode_h/y={}/h={:02}".format(year, (int(month) - 1) // 6 + 1)
        os.makedirs(date_path, exist_ok=True)
        with open("{}/data.txt".format(date_path), "a") as fp:
            if pubmed_data[str(key)][0][-1] in ['.', '?', '!']:
                combine_data = pubmed_data[key][0].replace('[', '').replace(']', '') + pubmed_data[key][1]
            else:
                combine_data = pubmed_data[key][0].replace('[', '').replace(']', '') + '.' + pubmed_data[key][1]

            fp.write(combine_data)
            fp.write('\n')

import json
import glob
from tqdm import tqdm


time_map = sorted(glob.glob("data_map/*.json"))

for file in tqdm(time_map):
    with open(file, 'r') as f:
        data_map = json.load(f)

    new_data_only_time = {}
    for pmid, value in data_map.items():
        new_data_only_time[pmid] = value["time"]

    with open('only_date_map/{}'.format(file.split('/')[-1]), 'w') as f:
        json.dump(new_data_only_time, f)

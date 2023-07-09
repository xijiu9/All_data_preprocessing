import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

with open("../Cooccurance_Matrix/all_time_map.json", "r") as f:
    all_time_map = json.load(f)

print("finish all time map")

with open("../Cooccurance_Matrix/PMID_search.json", "r") as f:
    PMID_dict = json.load(f)
# 创建一个初始值为0的矩阵

time_list_const = ['q_2019', 'q_2020_1', 'q_2020_2', 'q_2020_3', 'q_2020_4', 'q_2021_1', 'q_2021_2', 'q_2021_3', 'q_2021_4', 'q_2022_1', 'q_2022_2', 'q_2022_3', 'q_2022_4']

for time_before, time_after in zip(time_list_const[:-1], time_list_const[1:]):
    print(f"start loading {time_before}, {time_after}")
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

    matrix_previous = np.load(f'../Cooccurance_Matrix/co_occurance_matrix_{time_before}.npy')
    matrix_later = np.load(f'../Cooccurance_Matrix/co_occurance_matrix_{time_after}.npy')

    print("finish loading")
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

    cnt = 0
    geo_list, pmid_list = [], []
    for pmid in tqdm(all_time_map[f"{time_after}"]): # all_time_map["q_2022_1"] # pmid_2022_1
        subs, objs = PMID_dict[pmid]["sub"], PMID_dict[pmid]["obj"]
        if len(subs) * len(objs) == 0:
            continue
        feature = []
        for sub in subs:
            for obj in objs:
                feature.append(1 - matrix_previous[sub, obj] / matrix_later[sub, obj])
        feature = sorted(feature)
        # print(feature, '\n' * 5)
        sub_f = feature[-len(feature) // 20:]
        geo_mean = np.prod(sub_f)
        geo_list.append(geo_mean)

        if geo_mean > 0.5:
            pmid_list.append(pmid)

    with open(f"{time_after}_pmids.json", "w") as f:
        json.dump(pmid_list, f)

    plt.figure()
    plt.hist(geo_list, bins=100)
    plt.savefig(f"{time_after}_distribution.png")
    plt.close()

    print(cnt)

    del matrix_previous
    del matrix_later
    del pmid_list
    del geo_list

import json
import time
import numpy as np
import IPython
from tqdm import tqdm, trange

time_list_const = ['q_2019', 'q_2020_1', 'q_2020_2', 'q_2020_3', 'q_2020_4', 'q_2021_1', 'q_2021_2', 'q_2021_3', 'q_2021_4', 'q_2022_1', 'q_2022_2', 'q_2022_3', 'q_2022_4']

for time_before, time_after in zip(time_list_const[:-1], time_list_const[1:]):
    print(f"start loading {time_before}, {time_after}")
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

    matrix_previous = np.load(f'../Cooccurance_Matrix/co_occurance_matrix_{time_before}.npy')
    matrix_later = np.load(f'../Cooccurance_Matrix/co_occurance_matrix_{time_after}.npy')

    print("finish loading")
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

    indices = np.where(matrix_later > 0)
    i_indices = indices[0]
    j_indices = indices[1]

    nonzero_sample_list = []
    for i, j in zip(i_indices, j_indices):
        nonzero_sample_list.append([int(i), int(j)])
    print(nonzero_sample_list[:10])

    print(len(i_indices))
    print("finish where")
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

    with open(f"positive_indices_{time_after}.json", "w") as f:
        json.dump(nonzero_sample_list, f)

    sample_num = 1000000
    zeros_sample_list = []
    while len(zeros_sample_list) < sample_num:
        row, col = np.random.randint(0, matrix_later.shape[0]), np.random.randint(0, matrix_later.shape[1])
        if matrix_later[row, col] == 0:
            zeros_sample_list.append((row, col))

    with open(f"zero_indices_{time_after}_only_1000000.json", "w") as f:
        json.dump(zeros_sample_list, f)

    del matrix_previous
    del matrix_later
    del nonzero_sample_list
    del zeros_sample_list
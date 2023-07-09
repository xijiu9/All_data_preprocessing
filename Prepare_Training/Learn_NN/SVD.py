# import torch
# from transformers import (
#     AutoConfig,
#     AutoTokenizer,
#     AutoModelWithLMHead
# )
# import logging
# import time
# import argparse
# import os
# import json
# from datetime import datetime, timedelta
# from tqdm import tqdm, trange
# import torch.multiprocessing as mp
# import warnings
# import numpy as np
# import math
# import torch
# import matplotlib.pyplot as plt
# from scipy.stats import spearmanr
# import scipy
#
# print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading")
#
# matrix_2022_1 = np.load('../Cooccurance_Matrix/co_occurance_matrix_q_2022_1.npy')
# matrix_2021_4 = np.load('../Cooccurance_Matrix/co_occurance_matrix_q_2021_4.npy')
#
# print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start svd")
#
# def decompose(matrix, rank, scale):
#     U, sigma, V = scipy.sparse.linalg.svds(matrix, rank)
#     np.savez(f"svd_results_vertical_rank_{rank}_2022_1.npz", U=U, sigma=sigma, V=V)
#     print(f"save the svd_results_vertical_rank_{rank}_2022_1.npz")
#     print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading")
#
#     del U, sigma, V
#
# vertical_stack = np.vstack((matrix_2021_4, matrix_2022_1))
# # horizontal_stack = np.hstack((matrix_2021_4, matrix_2022_1))
#
# del matrix_2022_1, matrix_2021_4
# # import IPython
# # IPython.embed()
#
# decompose(vertical_stack, 10, 10000)
# decompose(vertical_stack, 100, 10000)
# decompose(vertical_stack, 1000, 10000)
#
# import IPython
# IPython.embed()

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelWithLMHead
)
import logging
import time
import argparse
import os
import json
from datetime import datetime, timedelta
from tqdm import tqdm, trange
import torch.multiprocessing as mp
import warnings
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import scipy

time_list_const = ['q_2022_3', 'q_2022_4']

for time_before, time_after in zip(time_list_const[:-1], time_list_const[1:]):
    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading")

    matrix_after = np.load(f'../Cooccurance_Matrix/co_occurance_matrix_{time_after}.npy')
    matrix_before = np.load(f'../Cooccurance_Matrix/co_occurance_matrix_{time_before}.npy')

    print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start svd")

    def decompose(matrix, rank, scale):
        U, sigma, V = scipy.sparse.linalg.svds(matrix, rank)
        np.savez(f"svd_results_vertical_rank_{rank}_{time_after}.npz", U=U, sigma=sigma, V=V)
        print(f"save the svd_results_vertical_rank_{rank}_{time_after}.npz")
        print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading")

        del U, sigma, V

    vertical_stack = np.vstack((matrix_before, matrix_after))
    # horizontal_stack = np.hstack((matrix_2021_4, matrix_2022_1))

    del matrix_before, matrix_after
    # import IPython
    # IPython.embed()

    decompose(vertical_stack, 10, 10000)
    decompose(vertical_stack, 100, 10000)
    decompose(vertical_stack, 1000, 10000)

    del vertical_stack

import IPython
IPython.embed()

import glob
import os
import json
import pandas as pd
import time
import nltk
from tqdm import tqdm
import matplotlib.pyplot as plt
import IPython
import numpy as np

import torch

import matplotlib.pyplot as plt

with open("/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Single_intersection/BioLAMA/wikidata/P780/Subject_dict.json", "r") as f:
    single_word_pmid = json.load(f)

import IPython
IPython.embed()

# covid_words = ["COVID-19",
#                "2019 nCoV Disease",
#                "2019-nCoV Disease",
#                "2019-nCoV Diseases",
#                "2019 nCoV Infection",
#                "2019-nCoV Infection",
#                "2019-nCoV Infections",
#                "2019 Novel Coronavirus Disease",
#                "2019 Novel Coronavirus Infection",
#                "Coronavirus Disease 19",
#                "Coronavirus Disease-19",
#                "Coronavirus Disease 2019",
#                "COVID 19",
#                "COVID 19 Pandemic",
#                "COVID-19 Pandemic",
#                "COVID-19 Pandemics",
#                "COVID 19 Virus Disease",
#                "COVID-19 Virus Disease",
#                "COVID-19 Virus Diseases",
#                "COVID 19 Virus Infection",
#                "Disease 2019, Coronavirus",
#                "Disease, 2019-nCoV",
#                "Disease, COVID-19 Virus",
#                "Infection, 2019-nCoV",
#                "Pandemic, COVID-19",
#                "SARS Coronavirus 2 Infection",
#                "SARS CoV 2 Infection",
#                "SARS-CoV-2 Infection",
#                "SARS-CoV-2 Infections",
#                "Virus Disease, COVID-19"
#                ]
# with open(
#         "/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Exact_intersection/BioLAMA/ctd/CD1"
#         "/Object_dict.json",
#         "r") as f:
#     Obj_dict = json.load(f)
#
# pmid_list = []
# for covid_word in covid_words:
#     covid_word = covid_word.lower()
#     pmid_list.extend(Obj_dict[covid_word])
#
# pmid_list = sorted(list(set(pmid_list)))
#
# with open("./Prepare_Training/Cooccurance_Matrix/all_time_map.json", "r") as f:
#     all_time_map = json.load(f)
#
# def intersection_of_lists(list1, list2):
#     # 将列表转换成集合
#     set1 = set(list1)
#     set2 = set(list2)
#
#     # 使用intersection()方法获取交集
#     intersection_result = set1.intersection(set2)
#
#     # 将交集转换回列表
#     result_list = list(intersection_result)
#     return result_list
#
# covid_time_map = {}
# time_list_const = ['q_2019', 'q_2020_1', 'q_2020_2', 'q_2020_3', 'q_2020_4', 'q_2021_1', 'q_2021_2', 'q_2021_3', 'q_2021_4', 'q_2022_1', 'q_2022_2', 'q_2022_3', 'q_2022_4', 'q_2023']
# for qtime in time_list_const:
#     covid_time_map[f"{qtime}"] = intersection_of_lists(all_time_map[f"{qtime}"], pmid_list)
#
# import IPython
# IPython.embed()


# def plot_line_length_distribution(file_path, save_path):
#     line_lengths = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             if len(line.strip()) < 100000:
#                 line_lengths.append(len(line.strip()))
#
#             if len(line.strip()) < 200:
#                 import IPython
#                 IPython.embed()
#
#     plt.hist(line_lengths, bins=100)  # 根据需要自定义 bins 的数量
#     plt.yscale('symlog')
#     plt.xlabel('Line Length')
#     plt.ylabel('Frequency')
#     plt.title('Line Length Distribution')
#     plt.savefig(save_path)  # 保存图像
#     plt.show()
#
#
#
# # 调用函数并传入txt文件路径和保存图像的路径
# txt_file_path = '/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Divide_Into_Months/training_data/mode_q/y=2020/q=02/data.txt'
# save_file_path = 'figure.png'  # 图像保存路径及文件名
# plot_line_length_distribution(txt_file_path, save_file_path)
#
# # file_path = "/m-ent1/ent1/xihc20/SHENG/Retrain_BioBERT/q_2020_1/pretrain/batch_mask/step_0_localrank0.pt"
# #

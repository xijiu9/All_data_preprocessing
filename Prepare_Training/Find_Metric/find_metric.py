import os
import re
import json
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# with open('feature_97.json', "r") as f:
#     feature = json.load(f)

# feature = feature[50:]
# 读取第一个 JSON 文件

def cal_MSE(data1_path, data2_path, character="mse"):
    # print(data1_path)
    # with open('control_probing_matrix_q_2022_1_LM_{M+N}.json', 'r') as file:
    # with open('control_probing_matrix_q_2022_1_LM_step_6001.json', 'r') as file:

    try:
        with open(data1_path, 'r') as file:
            data1 = json.load(file)
    except:
        return 0, 0

    # 读取第二个 JSON 文件
    with open(data2_path, 'r') as file:
        data2 = json.load(file)

    # print(data1_path, data2_path)
    # 提取概率值并转换为 NumPy 数组
    def get_P(data, log=False):
        prob = []

        if type(data) == dict:
            for sub, kv in data.items():
                for obj, v in kv.items():
                    # if matrix_2022_1[int(sub), int(obj)] > 0 and matrix_2021_4[int(sub), int(obj)] == 0:
                    # if [sub, obj] in feature:
                    # if len(prob) > 97:
                    #     break
                    if True:
                        if log:
                            prob.append(np.log10(v))
                            if v <= 0:
                                import IPython
                                IPython.embed()
                        else:
                            prob.append(v)
        elif type(data) == list:
            if log:
                prob = [kv[0] for kv in data]
            else:
                prob = [np.power(10, kv[0]) for kv in data]
        return np.array(prob)

    log = True
    probabilities1 = get_P(data1, log=log)
    probabilities2 = get_P(data2, log=log)

    sum_probabilities1 = np.sum(probabilities1)
    sum_probabilities2 = np.sum(probabilities2)

    normalized_probabilities1 = probabilities1 / sum_probabilities1
    normalized_probabilities2 = probabilities2 / sum_probabilities2

    # print(len(probabilities1), len(probabilities2))
    # 计算均方误差（MSE）
    if character == "mse":
        feature = np.mean((probabilities1 - probabilities2) ** 2)
    elif character == "normalized_mse":

        # 计算归一化后的均方误差
        feature = np.mean((normalized_probabilities1 - normalized_probabilities2) ** 2)

    # print("MSE:", mse)
    elif character == "ratio_mean":
        feature = np.median((probabilities1 / probabilities2))
    elif character == "normalized_ratio_mean":
        feature = np.median((normalized_probabilities1 / normalized_probabilities2))
    # print('ratio median:', ratio_median)
    elif character == "ratio_median":
        feature = np.mean((probabilities1 / probabilities2))
    elif character == "normalized_ratio_median":
        feature = np.mean((normalized_probabilities1 / normalized_probabilities2))
    # print('ratio mean:', ratio_median)
    elif character == 'wilxocon-W':
        W, p = stats.wilcoxon(probabilities1, probabilities2, alternative="greater")
        feature = W
    elif character == 'wilxocon-p':
        W, p = stats.wilcoxon(probabilities1, probabilities2, alternative="greater")
        feature = p
    elif character == 'normalized_wilxocon-W':
        W, p = stats.wilcoxon(normalized_probabilities1, normalized_probabilities2, alternative="greater")
        feature = W
    elif character == 'normalized_wilxocon-p':
        W, p = stats.wilcoxon(normalized_probabilities1, normalized_probabilities2, alternative="greater")
        feature = p
    elif character == 'spearman':
        rho, p_value = spearmanr(probabilities1, probabilities2)
        feature = rho
    #
    # def spearman_rho(list1, list2):
    #     rho, p_value = spearmanr(list1, list2)
    #     print("Spearman's秩相关系数: ", rho)
    #
    # spearman_rho(probabilities1, probabilities2)

    # print(stats.wilcoxon(probabilities1, probabilities2, alternative="greater"), '\n')

    return feature, re.split(r'[/.]', data1_path)[-2]


# time_features = ["q_2022_1", "q_2022_2", "q_2022_4"]
time_features = ["q_2022_2/GSAM_q_2022_2_only_10_epoch_rho_0.1_alpha_0.2"]

for character in ["mse", "normalized_mse", "ratio_mean", "normalized_ratio_mean",
                  "ratio_median", "normalized_ratio_median", "wilxocon-W", "wilxocon-p",
                  "normalized_wilxocon-W", "normalized_wilxocon-p", "spearman"]:
    for time_feature in time_features:
        metric_list, description_list = [], []
        for data_path1 in [f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_N+M.json',
                          f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_1001.json',
                          f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_2001.json',
                          f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_4001.json',
                          f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_6001.json',
                          f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_8001.json',
                          f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_10001.json',
                          f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_12001.json',
                          f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_16001.json',
                          f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_20001.json', #]:
                          f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_N.json']:

            # data_path2 = f'../Calculate_P/control_probing_matrix_{time_feature}/LM_step_N.json'
            data_path2 = f'../Calculate_P/control_probing_matrix_{time_feature}/Estimate_step_N.json'
            metric, description = cal_MSE(data_path1, data_path2, character=character)
            metric_list.append(metric)
            description_list.append(description)

        time_feature = time_feature.replace('/', '_')
        print(character, time_feature)
        print(metric_list)
        print('\n')

        plt.figure(figsize=(15, 5))
        last_value = metric_list[-1]  # 最后一个元素的值
        metric_list.pop()
        description_list.pop()
        while metric_list and metric_list[-1] == 0:
            metric_list.pop()
            description_list.pop()

        plt.plot(metric_list)  # 绘制列表除了最后一个元素的折线图
        plt.title(f"delete checkpoint {time_feature}")
        plt.xticks(range(len(metric_list)), description_list)
        plt.ylabel(f"{character}")
        plt.axhline(y=last_value, color='r', linestyle='--')  # 添加一条横向的虚线

        os.makedirs(f"{character}", exist_ok=True)
        plt.savefig(f"{character}/{time_feature}.png")
        plt.close()


